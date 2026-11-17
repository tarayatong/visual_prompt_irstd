"""Collect completed original-resolution evaluations without treating pending as zero."""
import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

from scripts.prepare_comparison import DEFAULT_OUTPUT, write_json
from visual_prompt.evaluation.comparison import refresh_full_image_fa


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--group", default="")
    parser.add_argument("--methods", nargs="+", default=["ours_frozen", "ours_joint", "HDNet", "HERA"])
    args = parser.parse_args()
    rows = []
    for dataset in ("IRSTD-1k", "NUDT-SIRST", "SIRST"):
        manifest = json.loads((args.root / dataset / "protocol/manifest.json").read_text(encoding="utf-8"))
        for method in args.methods:
            folder = args.root / args.group / dataset / method
            if args.group == "full16" and method in ("HERA", "ours_frozen", "ours_joint"):
                folder = args.root / dataset / method
            path = folder / "metrics/summary.json"
            row = dict(dataset=dataset, method=method, expected_test_images=len(manifest["test_ids"]),
                       predictions=len(list((folder / "predictions").glob("*.npy"))),
                       allocated_annotation_budget=50,
                       actual_annotation_budget=5 if method in ("ours_frozen", "HERA", "Matcher", "MPA", "ABCDFSS", "IFA", "TVGTANet", "PI-CLIP", "DSV-LFS") else 50,
                       training_epochs=None, convergence_epoch=None, schedule_status=None,
                       status="pending", IoU_percent=None, nIoU_percent=None, F1_percent=None,
                       Pd_percent=None, Fa_per_million=None)
            if path.exists():
                result = refresh_full_image_fa(json.loads(path.read_text(encoding="utf-8")))
                write_json(path, result)
                row["training_epochs"] = result.get("epochs")
                row["convergence_epoch"] = result.get("convergence_epoch")
                row["schedule_status"] = result.get("schedule_status")
                history_path = folder / "training_history.json"
                if row["training_epochs"] is None and history_path.exists():
                    row["training_epochs"] = len(json.loads(history_path.read_text(encoding="utf-8")))
                row["status"] = "complete" if result["complete_test"] else "partial"
                # Do not present a stale one-image smoke score alongside a later
                # in-progress full-test prediction count as a dataset result.
                if result["complete_test"]:
                    actual_ids = {p.stem for p in (folder / "predictions").glob("*.npy")}
                    if not set(manifest["test_ids"]).issubset(actual_ids):
                        raise ValueError(f"Incomplete prediction files for {dataset}/{method}")
                    if result.get("manifest_sha256") != manifest["manifest_sha256"]:
                        raise ValueError(f"Manifest mismatch for {dataset}/{method}")
                    for key in ("IoU_percent", "nIoU_percent", "F1_percent", "Pd_percent", "Fa_per_million"):
                        row[key] = result[key]
            rows.append(row)
    args.root = args.root / args.group
    args.root.mkdir(parents=True, exist_ok=True)
    write_json(args.root / "comparison_summary.json", rows)
    csv_path = args.root / "comparison_summary.csv"
    try:
        csv_file = csv_path.open("w", newline="", encoding="utf-8")
    except PermissionError:
        csv_path = args.root / f"comparison_summary_{datetime.now():%Y%m%d_%H%M%S}.csv"
        csv_file = csv_path.open("w", newline="", encoding="utf-8")
        print(f"Summary CSV is busy; saved updated table to {csv_path}", flush=True)
    with csv_file as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# 三数据集对比实验结果", "",
             f"更新时间：{datetime.now():%Y-%m-%d %H:%M:%S}（本机时间）。", "",
             "每个数据集固定从官方训练集随机选择 50 张，支持集为其中固定的 5 张；使用完整官方测试集。",
             "所有方法使用统一的原图分辨率评价接口。IoU、nIoU、F1、Pd 单位为 %，越高越好；Fa 为全图背景误分像素数除以全部图像像素数再乘1e6，越低越好。", "",
             "双 MLP 联合训练和 HDNet 实际使用 50 张标注；推理版本和 HERA 实际使用 5 张支持标注。HERA 包含支持集梯度适配。不同方法使用各自预训练与输入尺寸，详见各方法配置和 docs/comparison_execution.md。", "",
             "| 数据集 | 方法 | 实际标注数 | 训练轮数 | 状态 | IoU ↑ | nIoU ↑ | F1 ↑ | Pd ↑ | Fa ↓ |",
             "|---|---|---:|---:|---|---:|---:|---:|---:|---:|"]
    names = {"ours_frozen": "本方法：推理", "ours_joint": "本方法：双 MLP 联合训练", "HDNet": "HDNet", "HERA": "HERA"}
    for row in rows:
        scores = ["—" if row[k] is None else f"{row[k]:.2f}" for k in ("IoU_percent", "nIoU_percent", "F1_percent", "Pd_percent", "Fa_per_million")]
        state = {"complete": "完整测试完成", "partial": "部分测试", "pending": "尚无完整结果"}[row["status"]]
        lines.append(f"| {row['dataset']} | {names.get(row['method'], row['method'])} | {row['actual_annotation_budget']} | {row['training_epochs'] if row['training_epochs'] is not None else '—'} | {state} | " + " | ".join(scores) + " |")
    lines.extend(["", "## 相对双 MLP 联合训练的差值", "",
                  "下表为本方法双 MLP 联合训练减去对比方法。前四项正值表示本方法更高，Fa 负值表示本方法虚警更低。前三类精度及 Pd 差值为百分点。", "",
                  "| 数据集 | 对比方法 | ΔIoU | ΔnIoU | ΔF1 | ΔPd | ΔFa |",
                  "|---|---|---:|---:|---:|---:|---:|"])
    for dataset in ("IRSTD-1k", "NUDT-SIRST", "SIRST"):
        own = next((r for r in rows if r["dataset"] == dataset and r["method"] == "ours_joint"), None)
        if own is None:
            continue
        for method in ("HDNet", "HERA"):
            other = next((r for r in rows if r["dataset"] == dataset and r["method"] == method), None)
            if other is None:
                continue
            if own["status"] == other["status"] == "complete":
                differences = [f"{own[k] - other[k]:+.2f}" if own[k] is not None and other[k] is not None else "—"
                               for k in ("IoU_percent", "nIoU_percent", "F1_percent", "Pd_percent", "Fa_per_million")]
                lines.append(f"| {dataset} | {method} | " + " | ".join(differences) + " |")
    lines.extend(["", "仅完整测试结果进入对比表，未完成结果不记为零。各数据集/<方法>/metrics/ 保存逐图指标和完整精度汇总；predictions/ 保存原图尺寸预测，protocol/ 保存固定划分。", "",
                  "机器可读汇总：comparison_summary.json；CSV 若被其他应用占用，保存为带时间戳的备用文件。", ""])
    (args.root / "comparison_results.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
