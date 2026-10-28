"""CPU mathematics/contract checks; CUDA acceptance is a separate server gate."""
import unittest
import torch
from torch.nn import functional as F
from training.deterministic_point_sampling import deterministic_sample_point, sample_point


def native_sample(features, coordinates):
    add_dim = coordinates.ndim == 3
    if add_dim:
        coordinates = coordinates.unsqueeze(2)
    result = F.grid_sample(
        features, 2.0 * coordinates - 1.0, mode="bilinear",
        padding_mode="zeros", align_corners=False,
    )
    return result.squeeze(3) if add_dim else result


class DeterministicPointSamplingTests(unittest.TestCase):
    def compare(self, shape, coordinates, *, noncontiguous=False, dtype=torch.float64):
        generator = torch.Generator().manual_seed(351)
        features = torch.randn(shape, generator=generator, dtype=dtype)
        if noncontiguous:
            features = features.transpose(-1, -2).contiguous().transpose(-1, -2)
            self.assertFalse(features.is_contiguous())
        candidate_input = features.detach().requires_grad_(True)
        original_input = features.detach().requires_grad_(True)
        coordinates = coordinates.to(dtype)
        candidate = sample_point(candidate_input, coordinates, align_corners=False)
        original = native_sample(original_input, coordinates)
        self.assertEqual(candidate.shape, original.shape)
        self.assertEqual(candidate.dtype, original.dtype)
        self.assertTrue(torch.equal(candidate, original))
        upstream = torch.randn(candidate.shape, generator=generator, dtype=dtype)
        candidate.backward(upstream)
        original.backward(upstream)
        # Set before CUDA validation: fixed-tree vs native summation may differ.
        tolerance = 2e-12 if dtype == torch.float64 else 2e-5
        torch.testing.assert_close(
            candidate_input.grad, original_input.grad, rtol=tolerance, atol=tolerance,
        )
        return candidate_input.grad

    def test_alias_and_point_shape_multichannel(self):
        self.assertIs(sample_point, deterministic_sample_point)
        self.compare((2, 5, 7, 9), torch.rand(2, 37, 2, generator=torch.Generator().manual_seed(12)))

    def test_spatial_grid_shape(self):
        self.compare((2, 3, 7, 9), torch.rand(2, 3, 4, 2, generator=torch.Generator().manual_seed(13)))

    def test_boundary_repeated_and_out_of_range_coordinates(self):
        coords = torch.tensor([[
            [0., 0.], [1., 1.], [.5, .5], [.5, .5], [.5, .5],
            [-.05, .5], [1.05, .5], [.5, -.05], [.5, 1.05],
            [-2., -2.], [3., 3.], [.125, .125], [.875, .875],
        ]])
        self.compare((1, 4, 4, 4), coords)

    def test_noncontiguous_input_and_coordinates(self):
        coords = torch.rand(2, 2, 31, generator=torch.Generator().manual_seed(14)).transpose(1, 2)
        self.assertFalse(coords.is_contiguous())
        self.compare((2, 3, 7, 9), coords, noncontiguous=True)

    def test_single_pixel(self):
        coords = torch.tensor([[[0., 0.], [1., 1.], [.5, .5], [.2, .7], [2., 2.]]])
        self.compare((1, 3, 1, 1), coords)

    def test_empty_points(self):
        self.compare((2, 3, 7, 9), torch.empty(2, 0, 2))

    def test_real_point_count_12544(self):
        self.compare((2, 2, 24, 24), torch.rand(2, 12544, 2, generator=torch.Generator().manual_seed(15)))

    def test_float32_gradients(self):
        self.compare((2, 3, 13, 17), torch.rand(2, 121, 2, generator=torch.Generator().manual_seed(16)), dtype=torch.float32)

    def test_double_gradcheck(self):
        features = torch.randn(1, 2, 3, 4, dtype=torch.float64, requires_grad=True)
        coords = torch.tensor([[
            [.151, .237], [.623, .782], [.151, .237], [-.05, .45], [1.05, .61],
        ]], dtype=torch.float64)
        self.assertTrue(torch.autograd.gradcheck(
            lambda values: sample_point(values, coords), (features,),
            eps=1e-6, atol=1e-6, rtol=1e-4,
        ))

    def test_cpu_repetitions_are_bitwise_equal(self):
        coords = torch.full((2, 503, 2), .431, dtype=torch.float64)
        expected = self.compare((2, 2, 7, 9), coords)
        for _ in range(3):
            self.assertTrue(torch.equal(expected, self.compare((2, 2, 7, 9), coords)))

    def test_segment_reduction_does_not_subtract_global_cumsum(self):
        # A large earlier segment must not erase a small disjoint segment.
        features = torch.zeros(1, 1, 1, 2, dtype=torch.float64, requires_grad=True)
        coords = torch.tensor([[[.25, .5], [.25, .5], [.75, .5], [.75, .5]]], dtype=torch.float64)
        upstream = torch.tensor([[[1e20, 1e20, 1., 2.]]], dtype=torch.float64)
        sample_point(features, coords).backward(upstream)
        torch.testing.assert_close(
            features.grad, torch.tensor([[[[2e20, 3.]]]], dtype=torch.float64),
            rtol=0, atol=0,
        )

    def test_cpu_strict_mode(self):
        before = torch.are_deterministic_algorithms_enabled()
        warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        try:
            torch.use_deterministic_algorithms(True, warn_only=False)
            coords = torch.full((2, 31, 2), .431, dtype=torch.float64)
            self.compare((2, 2, 7, 9), coords)
        finally:
            torch.use_deterministic_algorithms(before, warn_only=warn_only)

    def test_forward_rng_is_not_consumed(self):
        features, coords = torch.ones(1, 3, 4, 4), torch.full((1, 11, 2), .431)
        before = torch.get_rng_state()
        sample_point(features, coords)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_cpu_autocast_preserves_native_policy(self):
        features, coords = torch.randn(1, 2, 4, 4, requires_grad=True), torch.rand(1, 17, 2)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            candidate, original = sample_point(features, coords), native_sample(features, coords)
        self.assertEqual(candidate.dtype, original.dtype)
        self.assertTrue(torch.equal(candidate, original))
        candidate.sum().backward()
        self.assertEqual(features.grad.dtype, features.dtype)

    def test_compile_outer_boundary_cpu(self):
        coords, features = torch.rand(1, 17, 2), torch.randn(1, 2, 4, 4, requires_grad=True)
        compiled = torch.compile(lambda x: sample_point(x, coords), backend="eager")
        candidate = compiled(features)
        self.assertTrue(torch.equal(candidate, native_sample(features, coords)))
        candidate.sum().backward()
        self.assertIsNotNone(features.grad)

    def test_coordinates_gradient_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Coordinate gradients"):
            sample_point(torch.ones(1, 1, 4, 4), torch.rand(1, 7, 2, requires_grad=True))

    def test_nonfinite_coordinates_are_rejected(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.assertRaisesRegex(ValueError, "finite"):
                sample_point(torch.ones(1, 1, 4, 4), torch.tensor([[[value, .5]]]))

    def test_unsupported_modes_fail_closed(self):
        for kwargs in (
            {"mode": "nearest"}, {"padding_mode": "border"},
            {"align_corners": True}, {"align_corners": None}, {"unknown": True},
        ):
            with self.assertRaises((ValueError, TypeError)):
                sample_point(torch.ones(1, 1, 4, 4), torch.rand(1, 7, 2), **kwargs)

    def test_input_contract(self):
        cases = (
            (torch.ones(1, 4, 4), torch.rand(1, 7, 2)),
            (torch.ones(1, 1, 4, 4), torch.rand(2, 7, 2)),
            (torch.ones(1, 1, 4, 4), torch.rand(1, 7, 3)),
            (torch.ones(1, 1, 4, 4, dtype=torch.int64), torch.rand(1, 7, 2)),
        )
        for features, coords in cases:
            with self.assertRaises((ValueError, TypeError)):
                sample_point(features, coords)


if __name__ == "__main__":
    unittest.main()
