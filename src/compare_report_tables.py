#!/usr/bin/env python3
"""
python compare_report_tables.py --metrics-dir report-metrics --output-dir report-metrics
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import pandas as pd


PAPER_VALUES = {
    "detection": {
        "F1": 0.902,
        "FPR": 0.016,
        "FNR": 0.146,
    },
    "ranking": {
        "nDCG_10": 0.707,
        "nDCG_100": 0.697,
        "nDCG": 0.931,
        "Pearson": 0.555,
        "Spearman": 0.457,
    },
}


def latex_table(df: pd.DataFrame, caption: str, label: str) -> str:
    latex = df.to_latex(index=False, float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else str(x))
    return latex.replace("\\begin{table}", f"\\begin{{table}}[h]\n\\caption{{{caption}}}\n\\label{{{label}}}")


def main():
    parser = argparse.ArgumentParser(description="Create final report-ready comparison tables.")
    parser.add_argument("--metrics-dir", type=Path, default=Path("report-metrics"))
    parser.add_argument("--output-dir", type=Path, default=Path("report-metrics"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    detection_path = args.metrics_dir / "detection_summary_overall.csv"
    ranking_path = args.metrics_dir / "ranking_summary_overall.csv"
    benchmark_path = args.metrics_dir / "benchmark_compare.csv"

    if not detection_path.exists():
        raise SystemExit(f"Missing {detection_path}")
    if not ranking_path.exists():
        raise SystemExit(f"Missing {ranking_path}")
    if not benchmark_path.exists():
        raise SystemExit(f"Missing {benchmark_path}")

    det = pd.read_csv(detection_path)
    rank = pd.read_csv(ranking_path)
    bench = pd.read_csv(benchmark_path)

    # Detection table
    det_rows = [{"implementation": "paper", **PAPER_VALUES["detection"]}]
    for impl in ["baseline", "adaptive"]:
        row = det[det["implementation"] == impl]
        if not row.empty:
            det_rows.append(
                {
                    "implementation": impl,
                    "F1": float(row.iloc[0]["F1"]),
                    "FPR": float(row.iloc[0]["FPR"]),
                    "FNR": float(row.iloc[0]["FNR"]),
                }
            )
    detection_table = pd.DataFrame(det_rows)

    # Ranking table using tstat method
    rank_rows = [{"implementation": "paper", **PAPER_VALUES["ranking"]}]
    for impl in ["baseline", "adaptive"]:
        row = rank[(rank["implementation"] == impl) & (rank["method"] == "tstat")]
        if not row.empty:
            rank_rows.append(
                {
                    "implementation": impl,
                    "nDCG_10": float(row.iloc[0]["nDCG_10"]),
                    "nDCG_100": float(row.iloc[0]["nDCG_100"]),
                    "nDCG": float(row.iloc[0]["nDCG"]),
                    "Pearson": float(row.iloc[0]["Pearson"]),
                    "Spearman": float(row.iloc[0]["Spearman"]),
                }
            )
    ranking_table = pd.DataFrame(rank_rows)

    # Efficiency table
    efficiency_cols = [
        "implementation",
        "total_runtime_sec",
        "peak_memory_mb",
        "number_of_monitored_subgroups",
        "total_artifact_bytes",
        "support_bucket_count",
    ]
    efficiency_table = bench[[c for c in efficiency_cols if c in bench.columns]].copy()

    # Derived ratios table
    ratio_cols = [
        "implementation",
        "subgroup_reduction_ratio",
        "runtime_reduction_ratio",
        "ranking_retention_ratio",
        "detection_retention_ratio",
    ]
    ratio_table = bench[[c for c in ratio_cols if c in bench.columns]].copy()

    # Save CSV
    detection_table.to_csv(args.output_dir / "final_detection_table.csv", index=False)
    ranking_table.to_csv(args.output_dir / "final_ranking_table.csv", index=False)
    efficiency_table.to_csv(args.output_dir / "final_efficiency_table.csv", index=False)
    ratio_table.to_csv(args.output_dir / "final_ratio_table.csv", index=False)

    # Save LaTeX
    latex_detection = latex_table(
        detection_table,
        caption="Detection comparison between paper, baseline, and adaptive implementations.",
        label="tab:detection_comparison",
    )
    latex_ranking = latex_table(
        ranking_table,
        caption="Ranking comparison between paper, baseline, and adaptive implementations.",
        label="tab:ranking_comparison",
    )
    latex_efficiency = latex_table(
        efficiency_table,
        caption="Efficiency comparison between baseline and adaptive implementations.",
        label="tab:efficiency_comparison",
    )
    latex_ratios = latex_table(
        ratio_table,
        caption="Derived ratio comparison for the adaptive implementation relative to baseline.",
        label="tab:ratio_comparison",
    )

    (args.output_dir / "final_detection_table.tex").write_text(latex_detection, encoding="utf-8")
    (args.output_dir / "final_ranking_table.tex").write_text(latex_ranking, encoding="utf-8")
    (args.output_dir / "final_efficiency_table.tex").write_text(latex_efficiency, encoding="utf-8")
    (args.output_dir / "final_ratio_table.tex").write_text(latex_ratios, encoding="utf-8")

    print("\n=== Final Detection Table ===")
    print(detection_table.to_string(index=False))
    print("\n=== Final Ranking Table ===")
    print(ranking_table.to_string(index=False))
    print("\n=== Final Efficiency Table ===")
    print(efficiency_table.to_string(index=False))
    print("\n=== Final Ratio Table ===")
    print(ratio_table.to_string(index=False))


if __name__ == "__main__":
    main()