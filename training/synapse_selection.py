"""Dependency-free scheduling, case aggregation and CSV for test-based selection."""
import csv
import math
from statistics import fmean


def should_evaluate(epoch, max_epochs, last_n_epochs):
    """All epochs are one-based; include exactly the final N completed epochs."""
    if not 1 <= last_n_epochs <= max_epochs:
        raise ValueError("last_n_epochs must be between 1 and max_epochs")
    return max_epochs - last_n_epochs < epoch <= max_epochs


def summarize_cases(results, expected_cases, foreground_classes):
    if len(results) != expected_cases or len({r["case"] for r in results}) != expected_cases:
        raise ValueError(f"Expected {expected_cases} distinct complete test cases")
    summary = {"num_cases": len(results)}
    for metric in ("dice", "hd95"):
        key = f"{metric}_per_class"
        for result in results:
            values = result[key]
            if len(values) != foreground_classes or not all(math.isfinite(v) for v in values):
                raise ValueError(f"Invalid {metric} values for {result['case']}")
        for i in range(foreground_classes):
            summary[f"{metric}_class_{i + 1}"] = fmean(r[key][i] for r in results)
        summary[f"mean_{metric}"] = fmean(
            summary[f"{metric}_class_{i + 1}"] for i in range(foreground_classes)
        )
    return summary


def append_csv(path, rows):
    if not rows:
        return
    fields = list(rows[0])
    exists = path.is_file() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)
