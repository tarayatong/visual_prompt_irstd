"""Verify follow-up experiment results and generate a review summary."""

import argparse
import csv
import json
import math
from pathlib import Path
import statistics

from scripts.prepare_comparison import DEFAULT_OUTPUT
from scripts.run_core_ir_common_protocol import NATIVE_FULL_SIZES
from scripts.run_core_ir_followup_queue import DOMAINS, OUT, REPRESENTATIVES, ROOT, TRANSFER_DOMAINS
from visual_prompt.evaluation.comparison import summarize as summarize_counts

SUMMARY = Path("epoch_0120/metrics/summary.json")
TRAIN_SEEDS = (111, 112, 113)
SEED_RUN = ROOT / "outputs/core_ir_final_three_seeds_20261116"
SUPPORT_METRICS = {
    "IoU_percent": "IoU (%)", "nIoU_percent": "nIoU (%)", "F1_percent": "F1 (%)",
    "Pd_percent": "Pd (%)", "Fa_per_million": "Fa (/百万像素)",
}
PAPER_METHOD_ORDER = (
    "ISNet", "DNANet", "SCTransNet", "MSHNet", "MSHNet+NS-FPN", "ILNet", "HDNet",
    "RPCANet", "SAM-SPL", "IRSAM", "IRPruneDet", "MLP-Net", "SAMamba", "IR-SAM2",
    "MPCNet", "SANSA", "Ours",
)
PAPER_COMMENTED_METHODS = {"ISNet", "RPCANet", "IRPruneDet", "MLP-Net", "MPCNet"}
COMPLEXITY_HEADER = "Method | Encoder input | Trainable (M) | GFLOPs | IoU (%)"


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def table(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def verified_iou(summary, test_ids):
    row = load(summary)
    measurements = table(summary.with_name("per_image.csv"))
    if [record["sample_id"] for record in measurements] != test_ids:
        raise ValueError(f"Unexpected test coverage in {summary}")
    tp, fp, fn = (sum(int(record[key]) for record in measurements) for key in ("tp", "fp", "fn"))
    iou = 100 * tp / (tp + fp + fn) if tp + fp + fn else 100.
    if (row["images"] != len(test_ids) or not row["complete_test"] or
        not math.isclose(iou, row["IoU_percent"], abs_tol=1e-9)):
        raise ValueError(f"IoU or evaluation metadata disagree with per-image records in {summary}")
    return iou


def metric(value):
    return f"{value:.2f}" if value is not None else "—"


def verified_support_metrics(path, manifest, seed, checkpoint_sha, support_ids, draw=None):
    row = load(path)
    counts = table(path.with_name("per_image.csv"))
    if [record["sample_id"] for record in counts] != manifest["test_ids"]:
        raise ValueError(f"Unexpected test coverage in {path}")
    if (not row["complete_test"] or not row["prediction_coverage_verified"] or
        row["images"] != len(counts) or row["manifest_sha256"] != manifest["manifest_sha256"] or
        row["epochs"] != 120 or row["seed"] != seed or row["checkpoint_sha256"] != checkpoint_sha or
        row.get("support_ids", row.get("support_annotation_ids")) != support_ids or
        row.get("support_draw_seed", draw) != draw):
        raise ValueError(f"Unexpected support evaluation protocol in {path}")
    metrics = summarize_counts([
        {key: float(value) if key == "image_iou" else int(value)
         for key, value in record.items() if key != "sample_id"}
        for record in counts
    ])
    for key in SUPPORT_METRICS:
        if not math.isclose(metrics[key], row[key], rel_tol=0, abs_tol=1e-9):
            raise ValueError(f"{key} disagrees with per-image records in {path}")
    return {key: metrics[key] for key in SUPPORT_METRICS}


def support_comparison(manifests, missing):
    fixed_csv = SEED_RUN / "fixed_support/runs.csv"
    fixed = {(r["dataset"], int(r["seed"])): r for r in table(fixed_csv)} if fixed_csv.is_file() else {}
    prior_csv = SEED_RUN / "runs.csv"
    archived = {(r["dataset"], int(r["seed"]), int(r["support_draw_seed"])): ROOT / r["summary"]
                for r in table(prior_csv) if r.get("support_draw_seed")} if prior_csv.is_file() else {}
    random_protocol = load(SEED_RUN / "protocol.json")["datasets"]
    modes = ("固定 5 张（主表）", "随机抽 5 张（每组用于整个测试集）")
    header = ["训练支持", "推理支持"]
    rows = {mode: ["每个 episode 从 train50 随机抽 1 张", mode] for mode in modes}
    for dataset in DOMAINS:
        manifest = manifests[dataset]
        draws = random_protocol[dataset]["inference_support_sets"]
        header.extend(f"{dataset} {label}" for label in SUPPORT_METRICS.values())
        scores = {mode: [] for mode in modes}
        for seed in TRAIN_SEEDS:
            record = fixed.get((dataset, seed))
            if record is None or not (ROOT / record["summary"]).is_file():
                missing.append(f"support/{dataset}/seed{seed}/fixed")
                continue
            checkpoint_sha = record["checkpoint_sha256"]
            fixed_scores = verified_support_metrics(ROOT / record["summary"], manifest, seed,
                                                    checkpoint_sha, manifest["support_ids"])
            if any(not math.isclose(value, float(record[key]), rel_tol=0, abs_tol=1e-9)
                   for key, value in fixed_scores.items()):
                raise ValueError(f"Fixed-support metrics disagree with {fixed_csv} for {dataset}/seed{seed}")
            scores[modes[0]].append(fixed_scores)
            random_scores = []
            for draw_id, ids in draws.items():
                draw = int(draw_id)
                leaf = OUT / "support" / dataset / f"seed{seed}" / f"support{draw}" / SUMMARY
                path = leaf if leaf.is_file() else archived.get((dataset, seed, draw))
                if path is None or not path.is_file():
                    missing.append(f"support/{dataset}/seed{seed}/support{draw}")
                    continue
                random_scores.append(verified_support_metrics(path, manifest, seed, checkpoint_sha, ids, draw))
            if len(random_scores) == len(draws) == 3:
                scores[modes[1]].append({key: statistics.mean(r[key] for r in random_scores)
                                         for key in SUPPORT_METRICS})
        for mode, records in scores.items():
            for key in SUPPORT_METRICS:
                values = [r[key] for r in records]
                rows[mode].append(f"{statistics.mean(values):.2f} ± {statistics.stdev(values):.2f}"
                                  if len(values) == len(TRAIN_SEEDS) else "—")
    return [
        "支持集对照（训练种子 111、112、113，均值 ± 样本标准差）",
        " | ".join(header),
        *(" | ".join(row) for row in rows.values()),
        "两行使用相同的第 120 轮权重。随机推理在每个数据集上抽取 3 组支持，三个训练种子共用同样的支持组。",
        "先对每个训练种子的 3 次完整测试取均值，再跨训练种子计算均值与样本标准差。固定 5 张仅用于推理。",
    ]


def summarize():
    status = load(OUT / "queue.json")
    manifests = {d: load(DEFAULT_OUTPUT / d / "protocol/manifest.json") for d in DOMAINS}
    lines = ["CoRe-IR 补充实验核验汇总", f"队列状态  {status['status']}", ""]
    missing = []
    points = {}
    paper_methods = tuple(method for method in PAPER_METHOD_ORDER if method not in PAPER_COMMENTED_METHODS)
    commented_methods = tuple(method for method in PAPER_METHOD_ORDER if method in PAPER_COMMENTED_METHODS)
    representatives = tuple(method for method in PAPER_METHOD_ORDER if method in REPRESENTATIVES)
    native_methods = sorted(NATIVE_FULL_SIZES, key=PAPER_METHOD_ORDER.index)
    lines.extend((
        "IRSTD-1k 模型复杂度与 IoU（50 张训练标注）",
        "Encoder input 为测算 GFLOPs 的内部尺寸，GFLOPs 按单张查询报告。",
        "总参数量和五张支持图像的 GFLOPs 见同目录 complexity.csv。",
        "",
    ))
    for phase, title, groups in (
        ("complexity", "外部输入 640×640（LaTeX 表显示的方法）", [("IRSTD-1k", 640, paper_methods)]),
        ("complexity", "外部输入 640×640（LaTeX 表注释行中的方法）", [("IRSTD-1k", 640, commented_methods)]),
        ("complexity", "外部输入 256×256", [("IRSTD-1k", 256, representatives)]),
        ("full_native", "全量训练（各方法的训练配方）", [
            (dataset, NATIVE_FULL_SIZES[method], (method,)) for dataset in DOMAINS for method in native_methods
        ]),
    ):
        native = phase == "full_native"
        header = COMPLEXITY_HEADER
        if native:
            header += " | 数据集 | 外部尺寸 | 标签数 | 总参数 M | 五支持 GFLOPs | 轮数 | 优化器 | 学习率 | 有效 batch | 微批大小 | 精度"
        lines.extend((title, header))
        for dataset, size, methods in groups:
            manifest = manifests[dataset]
            for method in methods:
                folder = OUT / phase / dataset / str(size) / method
                result = folder / "result.json"
                if not result.is_file():
                    missing.append(f"{phase}/{dataset}/{size}/{method}")
                    continue
                row = load(result)
                epochs = 600 if native and method == "ILNet" else 120
                if row["training_horizon"] != epochs or (native and row.get("training_protocol") != "method_specific_full"):
                    raise ValueError(f"Unexpected training horizon or recipe in {result}")
                metrics = Path(f"epoch_{epochs:04d}/metrics/summary.json")
                ious = [verified_iou(folder / f"support_{index}" / metrics, manifest["test_ids"])
                        for index, _ in enumerate(row["support_draws"])]
                iou = statistics.mean(ious)
                annotations = manifest["source_train_ids" if native else "train_ids"]
                if (row["method"], row["dataset"], row["input_size"], row["train_annotation_ids"]) != (
                    method, dataset, size, annotations
                ) or not math.isclose(iou, row["IoU_percent"], abs_tol=1e-9):
                    raise ValueError(f"Training protocol or support average mismatch in {result}")
                points[phase, dataset, size, method] = (iou, row["GFLOPs"])
                encoder_size = row["backbone_input_size"]
                values = (
                    method, f"{encoder_size}×{encoder_size}", f"{row['parameters_trainable'] / 1e6:.3f}",
                    metric(row["GFLOPs"]), metric(iou),
                )
                if native:
                    values += (
                        dataset, f"{size}×{size}", len(annotations), metric(row["parameters_total"] / 1e6),
                        metric(row["support_GFLOPs"]), epochs, row["optimizer"], row["learning_rate"],
                        row["batch_size"], row["microbatch"], row["precision"],
                    )
                lines.append(" | ".join(map(str, values)))
        lines.append("")
    lines.extend(("分辨率配对", "方法 | IoU 256 | IoU 640 | IoU 差（百分点） | GFLOPs 256 | GFLOPs 640"))
    for method in representatives:
        low = points.get(("complexity", "IRSTD-1k", 256, method))
        high = points.get(("complexity", "IRSTD-1k", 640, method))
        if low and high:
            lines.append(" | ".join([method, *map(metric, (low[0], high[0], high[0] - low[0], low[1], high[1]))]))
    lines.extend(("", *support_comparison(manifests, missing)))
    lines.extend(("", "外域泛化", "目标数据集 | 方法 | 直接迁移 IoU | 仅换五支持 IoU | 五张微调 IoU | 微调相对直接（百分点）"))
    for dataset in TRANSFER_DOMAINS:
        root = OUT / "transfer" / dataset
        protocol = root / "protocol.json"
        if not protocol.is_file():
            missing.append(f"transfer/{dataset}/protocol")
            continue
        test_ids = load(protocol)["test_ids"]
        for method in ("ILNet", "SANSA", "Ours"):
            conditions = ("direct", "update5") if method == "ILNet" else ("direct", "support5", "update5")
            values = {}
            for condition in conditions:
                summary = root / method / condition / SUMMARY
                if summary.is_file():
                    values[condition] = verified_iou(summary, test_ids)
                else:
                    missing.append(f"transfer/{dataset}/{method}/{condition}")
            if values:
                delta = values["update5"] - values["direct"] if "update5" in values and "direct" in values else None
                lines.append(" | ".join([dataset, method, *map(metric, (
                    values.get("direct"), values.get("support5"), values.get("update5"), delta))]))
    if missing:
        lines.extend(("", "待补结果", *missing))
    lines.append("")
    return lines, missing


def write_report(partial=False):
    rows, missing = summarize()
    if missing and not partial:
        raise ValueError(f"{len(missing)} result groups are pending")
    report = OUT / ("verified_partial.txt" if missing else "verified_report.txt")
    report.write_text("\n".join(rows), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partial", action="store_true", help="Allow an explicitly incomplete snapshot.")
    args = parser.parse_args()
    print(write_report(partial=args.partial))
