"""Collect completed ablation summaries into one comparison table."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


FIELDS = (
    "setting",
    "use_vector_correction",
    "use_high_freq_negative",
    "use_bright_negative",
    "use_sliding_window",
    "window_size",
    "mask_score_mode",
    "mIoU_percent",
    "F1_percent",
    "Pd_percent",
    "Fa_per_million",
    "sim_top1_hit_percent",
    "sim_top1_hit_dilate3_percent",
)


def parse_args():
    parser = argparse.ArgumentParser("Collect sliding-window ablation summaries.")
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", default=None, type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    root = args.root.resolve()
    output = args.output.resolve() if args.output else root / "ablation_summary.csv"
    rows = []
    for path in sorted(root.rglob("sliding_window_semantic_mask_summary.csv")):
        with path.open("r", encoding="utf-8", newline="") as handle:
            source = next(csv.DictReader(handle))
        relative = path.relative_to(root)
        setting = relative.parts[0] if len(relative.parts) > 1 else path.parent.name
        row = {field: source.get(field, "") for field in FIELDS}
        row["setting"] = setting
        rows.append(row)
    if not rows:
        raise RuntimeError(f"No completed summaries found under {root}")

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(output)


if __name__ == "__main__":
    main()

