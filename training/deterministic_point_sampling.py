"""Deterministic input gradients for the isolated point-sampling protocol.

The forward is native bilinear/zero-padding grid_sample. The backward stably
sorts four-neighbour contributions by destination pixel, reduces each segment
by a fixed tree, and writes once per destination. No atomic addition is used.
This is an explicit eager boundary inside an otherwise compiled model.
"""
from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


class _DeterministicBilinearInput(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_features: Tensor, grid: Tensor) -> Tensor:
        # Function.forward does not record the native nondeterministic backward.
        # Its autocast policy and output dtype remain those of grid_sample.
        output = F.grid_sample(
            input_features, grid, mode="bilinear",
            padding_mode="zeros", align_corners=False,
        )
        ctx.save_for_backward(grid)
        ctx.input_shape = tuple(input_features.shape)
        ctx.input_dtype = input_features.dtype
        ctx.accumulation_dtype = (
            torch.float64 if output.dtype == torch.float64 else torch.float32
        )
        return output

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        (grid,) = ctx.saved_tensors
        n, channels, height, width = ctx.input_shape
        spatial_size = height * width
        points = grid.shape[1] * grid.shape[2]
        if points == 0:
            return grad_output.new_zeros(ctx.input_shape, dtype=ctx.input_dtype), None

        # CUDA opmath is float for half/bfloat16. The saved grid already
        # includes the original coordinate mapping and its dtype rounding.
        work_grid = grid.to(ctx.accumulation_dtype).reshape(n, points, 2)
        x = ((work_grid[..., 0] + 1) * width - 1) / 2
        y = ((work_grid[..., 1] + 1) * height - 1) / 2
        x0, y0 = torch.floor(x), torch.floor(y)
        x1, y1 = x0 + 1, y0 + 1
        neighbours = (
            (x0, y0, (x1 - x) * (y1 - y)),
            (x1, y0, (x - x0) * (y1 - y)),
            (x0, y1, (x1 - x) * (y - y0)),
            (x1, y1, (x - x0) * (y - y0)),
        )
        indices, weights = [], []
        for neighbour_x, neighbour_y, weight in neighbours:
            valid = (
                (neighbour_x >= 0) & (neighbour_x < width)
                & (neighbour_y >= 0) & (neighbour_y < height)
            )
            # Clamp before int conversion avoids overflowing int64 for very
            # large finite zero-padding coordinates.
            pixel = (
                neighbour_y.clamp(0, height - 1).to(torch.long) * width
                + neighbour_x.clamp(0, width - 1).to(torch.long)
            )
            indices.append(torch.where(valid, pixel, spatial_size))
            weights.append(torch.where(valid, weight, 0))

        destination = torch.cat(indices, dim=1)
        weights = torch.cat(weights, dim=1)
        # Stable order: NW, NE, SW, SE, then original point order per neighbour.
        destination, order = torch.sort(destination, dim=1, stable=True)
        values = (
            grad_output.to(ctx.accumulation_dtype).reshape(n, channels, points)
            .repeat(1, 1, 4) * weights[:, None, :]
        )
        values = torch.gather(values, 2, order[:, None, :].expand(-1, channels, -1))

        # Segmented doubling scan has fixed additions with one writer. It does
        # not subtract global cumsums or atomically scatter repeated indices.
        offset, count = 1, destination.shape[1]
        while offset < count:
            same_segment = destination[:, offset:] == destination[:, :-offset]
            addition = torch.where(
                same_segment[:, None, :], values[:, :, :-offset], 0,
            )
            values = values + F.pad(addition, (offset, 0))
            offset *= 2

        segment_end = torch.ones_like(destination, dtype=torch.bool)
        segment_end[:, :-1] = destination[:, :-1] != destination[:, 1:]
        segment_end &= destination < spatial_size
        keep = segment_end[:, None, :].expand(-1, channels, -1)
        offsets = (
            torch.arange(n * channels, device=grid.device, dtype=torch.long)
            .reshape(n, channels, 1) * spatial_size
        )
        unique_destinations = (destination[:, None, :] + offsets)[keep]
        reduced_values = values[keep]
        gradient = torch.zeros(
            n * channels * spatial_size, device=grad_output.device,
            dtype=ctx.accumulation_dtype,
        )
        # Segment ends yield unique destinations; batch/channel offsets are
        # disjoint. index_copy therefore has exactly one writer per pixel.
        gradient.index_copy_(0, unique_destinations, reduced_values)
        return gradient.reshape(ctx.input_shape).to(ctx.input_dtype), None


@torch.compiler.disable
def deterministic_sample_point(
    input_features: Tensor, point_coordinates: Tensor, add_dim: bool = False, **kwargs,
) -> Tensor:
    """Sample NP2/NHW2 finite coordinates, differentiating only NCHW input.

    Supports bilinear/zeros/align_corners=False only. Finite out-of-range
    coordinates retain native zero padding. Coordinate gradients are rejected.
    Native forward math, autocast and output dtype are retained. The local eager
    boundary prevents compilation from reassociating the prescribed reduction.
    """
    mode = kwargs.pop("mode", "bilinear")
    padding_mode = kwargs.pop("padding_mode", "zeros")
    align_corners = kwargs.pop("align_corners", False)
    if kwargs:
        raise TypeError(f"Unsupported point-sampling arguments: {sorted(kwargs)}")
    if mode != "bilinear" or padding_mode != "zeros" or align_corners is not False:
        raise ValueError("Deterministic sampling requires bilinear/zeros/align_corners=False.")
    if input_features.ndim != 4 or point_coordinates.ndim not in (3, 4):
        raise ValueError("Expected NCHW features and NP2 or NHW2 coordinates.")
    if point_coordinates.shape[0] != input_features.shape[0] or point_coordinates.shape[-1] != 2:
        raise ValueError("Coordinate batch must match features and its last dimension must be 2.")
    if not input_features.is_floating_point() or not point_coordinates.is_floating_point():
        raise TypeError("Features and coordinates must be floating-point tensors.")
    if input_features.device != point_coordinates.device:
        raise ValueError("Features and coordinates must be on the same device.")
    if point_coordinates.requires_grad:
        raise ValueError("Coordinate gradients are unsupported; criterion coordinates must not require grad.")
    if not bool(torch.isfinite(point_coordinates).all()):
        raise ValueError("Point coordinates must be finite.")
    if any(size <= 0 for size in input_features.shape):
        raise ValueError("Feature dimensions must be positive.")
    if point_coordinates.ndim == 3:
        add_dim = True
        point_coordinates = point_coordinates.unsqueeze(2)
    output = _DeterministicBilinearInput.apply(
        input_features, 2.0 * point_coordinates - 1.0,
    )
    return output.squeeze(3) if add_dim else output


# Local drop-in import, never a global Transformers monkey patch.
sample_point = deterministic_sample_point
