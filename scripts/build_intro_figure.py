"""Collect the prior transfer results and measure held-out SAM2 cosine similarity."""

import argparse
import shutil

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import torch

from scripts import run_motivation_current_ours as prior
from visual_prompt.core.semantic_direction import (
    SemanticDirectionConfig,
    bright_flat_mask_from_target_mean,
    build_score_token,
    edge_mask_from_image,
    mean_token,
    pool_feature_token,
)
from visual_prompt.models.tiny_shared_memory import (
    build_episode_keys,
    descriptor,
    support_regions,
)

OUT = prior.ROOT / "outputs/intro_figure1"
ARCHIVE = prior.ROOT / "outputs/nudt_irstd_shot_curve_20261112/first_figure_ten_methods"
DOMAIN = "IRSTD-1k"
GRID = 64
SIZE = 640
TILE = 8
REGIONS = ("Target", "High frequency", "High value")
COLORS = ("#068264", "#c74357", "#b47b08")


def save_figure(fig, stem):
    for suffix in ("pdf", "png"):
        fig.savefig(OUT / f"{stem}.{suffix}", dpi=240)
    plt.close(fig)


def transfer_and_adaptation():
    archived = ARCHIVE / "first_figure_ten_methods.csv"
    rows = prior.csv_rows(archived)
    protocol = prior.read_json(ARCHIVE / "protocol.json")
    assert prior.read_json(ARCHIVE / "audit.json")["status"] == "passed"
    assert len(rows) == 50 and len(protocol["source_test_ids"]) == 664
    assert len(protocol["test_ids"]) == 201
    shots = 5
    assert len(protocol["shot_ids"][str(shots)]) == shots
    adapted = {row["method"]: float(row["target_iou"]) for row in rows if int(row["shot"]) == shots}
    transfer = []
    for row in rows:
        if int(row["shot"]) == 0:
            source, target = (float(row[f"{domain}_iou"]) for domain in ("source", "target"))
            method = row["method"]
            after = adapted[method]
            transfer.append(dict(method=method, source_iou=source, target_iou=target,
                                 drop_pp=source - target, adaptation_shots=shots,
                                 adapted_target_iou=after, adaptation_gain_pp=after - target))
    transfer.sort(key=lambda row: row["source_iou"], reverse=True)
    assert len(transfer) == 10
    prior.save_csv(OUT / "fig1a_cross_domain.csv", transfer)

    fig, ax = plt.subplots(figsize=(6.6, 4.45), layout="constrained")
    green, warm, neutral = "#08724e", "#dca081", "#7e8790"
    for i, row in enumerate(transfer):
        target, source, after = row["target_iou"], row["source_iou"], row["adapted_target_iou"]
        ours = row["method"] == "Ours"
        color = green if ours else warm
        ax.plot([target, source], [i, i], ":", color="#b9c0c6", lw=1.4, zorder=1)
        ax.plot([target, after], [i, i], color=color, lw=3.0 if ours else 2.0, zorder=2)
        ax.scatter(target, i, s=33, marker="o", facecolors="white", edgecolors=color,
                   linewidths=1.5, zorder=3)
        ax.scatter(after, i, s=40 if ours else 30, marker="D", color=color, zorder=3)
        ax.scatter(source, i, s=28, marker="s", color=neutral, zorder=3)
        ax.text(83, i, f"{row['adaptation_gain_pp']:+.1f}", va="center", ha="center",
                color=green if ours else "#725e53", fontsize=9, fontweight="bold" if ours else "normal")
        if ours:
            for value in (target, after):
                ax.annotate(f"{value:.2f}", (value, i), xytext=(0, 7), textcoords="offset points",
                            ha="center", va="bottom", color=green, fontsize=8.5, fontweight="bold")
    ax.set(title="NUDT-SIRST → IRSTD-1k", xlabel="Test IoU (%)", xlim=(0, 91),
           yticks=range(len(transfer)),
           yticklabels=["CoRe-IR" if r["method"] == "Ours" else r["method"] for r in transfer],
           ylim=(9.65, -1.35), xticks=range(0, 71, 10))
    ax.text(83, -1.0, "Adaptation\nΔ IoU (pp)", ha="center", va="center", color="#725e53", fontsize=8.5)
    ax.get_yticklabels()[0].set_color(green)
    ax.get_yticklabels()[0].set_fontweight("bold")
    ax.set_axisbelow(True)
    ax.grid(axis="x", alpha=.2)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(handles=[
        Line2D([], [], marker="s", ls="", color=neutral, label="Source"),
        Line2D([], [], marker="o", ls="", color="#725e53", mfc="white", label="Target · 0-shot"),
        Line2D([], [], marker="D", ls="", color="#725e53", label=f"Target · {shots}-shot / 100 updates"),
    ], loc="upper center", bbox_to_anchor=(.5, -.13), ncols=3, frameon=False, fontsize=8.5)
    save_figure(fig, "fig1a_cross_domain")

    for suffix in ("pdf", "png", "csv"):
        shutil.copyfile(ARCHIVE / f"first_figure_ten_methods.{suffix}",
                        OUT / f"fig1c_adaptation_retention.{suffix}")


def tile_for(region):
    """Select the tile with most eligible descriptor cells, breaking ties in raster order."""
    cells = region.cpu().numpy()
    blocks = cells.reshape(GRID // TILE, TILE, GRID // TILE, TILE)
    counts = blocks.sum(axis=(1, 3))
    row, col = np.unravel_index(counts.argmax(), counts.shape)
    if not counts[row, col]:
        return None, None
    y, x = int(row * TILE), int(col * TILE)
    selected = torch.zeros_like(region)
    selected[y:y + TILE, x:x + TILE] = region[y:y + TILE, x:x + TILE]
    stride = SIZE // GRID
    return selected, (stride * x, stride * y, stride * TILE, stride * TILE)


def query_regions(image, mask):
    geometry = support_regions(image, mask, GRID)
    _, labels, stats, centers = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    largest = int(stats[1:, cv2.CC_STAT_AREA].argmax())
    target_region = geometry["instances"][largest]
    foreground = torch.stack(geometry["instances"]).any(0)
    hx, hy = centers[largest + 1]
    width = TILE * (SIZE // GRID)
    x0, y0 = np.clip(np.rint([hx - width / 2, hy - width / 2]).astype(int), 0, SIZE - width)
    config = SemanticDirectionConfig()
    excluded = cv2.dilate(mask.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    high_frequency = edge_mask_from_image(image, config.edge_method, config.edge_quantile) & ~excluded
    high_value = bright_flat_mask_from_target_mean(image, mask) & ~mask
    regions = [(target_region, (int(x0), int(y0), width, width), labels == largest + 1)]
    for region, eligible in zip(geometry["negatives"], (high_frequency, high_value)):
        selected, box = tile_for(region & ~foreground)
        if selected is None:
            regions.append((None, None, None))
        else:
            pixel_cells = np.repeat(np.repeat(selected.numpy(), SIZE // GRID, axis=0), SIZE // GRID, axis=1)
            regions.append((selected, box, pixel_cells & eligible))
    return regions


@torch.inference_mode()
def measure():
    manifest = prior.bg.manifest(DOMAIN)
    engine = prior.Engine(DOMAIN, prior.selected()[DOMAIN])
    assert not set(engine.support_ids) & set(manifest["test_ids"])
    inputs, _, support_geometry = zip(*(engine.support[sid] for sid in engine.support_ids))
    backbone = engine.backbone(torch.stack(inputs)[None])
    keys = build_episode_keys(engine.model, backbone, range(len(inputs)), support_geometry)
    positives, negatives = [], {"high_freq": [], "bright_flat": []}
    for i, geometry in enumerate(support_geometry):
        feature = descriptor(engine.model, backbone, i)[0].permute(1, 2, 0)
        positives.append(mean_token([
            pool_feature_token(feature, region.to(feature.device), "mean_max")
            for region in geometry["instances"]
        ]))
        for name, region in zip(negatives, geometry["negatives"]):
            token = pool_feature_token(feature, region.to(feature.device), "mean")
            if token is not None:
                negatives[name].append(token)
    corrected, info = build_score_token(positives, negatives, SemanticDirectionConfig())
    assert torch.allclose(keys[0], mean_token(positives)[0], atol=1e-6)
    assert torch.allclose(keys[1], corrected[0], atol=1e-6)
    assert info["negative_rank"] == 2 and max(map(abs, info["score_negative_cosines"])) < 1e-5
    del backbone

    rows, example = [], None
    for n, sid in enumerate(manifest["test_ids"], 1):
        image, mask = prior.base.raw_pair(manifest, sid, SIZE, True)
        image = image.astype(np.uint8)
        regions = query_regions(image, mask > 0)
        backbone = engine.backbone(prior.tensor(image)[None, None])
        feature = descriptor(engine.model, backbone, 0)[0].permute(1, 2, 0)
        for name, (region, box, pixels) in zip(REGIONS, regions):
            if region is None:
                continue
            mode = "mean_max" if name == "Target" else "mean"
            token = pool_feature_token(feature, region.to(feature.device), mode)
            before, after = (token @ keys[:2].T)[0].cpu().tolist()
            rows.append(dict(sample_id=sid, region=name, before=before, after=after,
                             cells=int(region.sum()), pixels=int(pixels.sum()),
                             x=box[0], y=box[1], width=box[2], height=box[3]))
        if example is None and all(r[0] is not None for r in regions) and regions[1][1] != regions[2][1]:
            example = sid
        if n % 25 == 0:
            print(f"Encoded {n} held-out images", flush=True)
    assert example
    assert sum(row["region"] == "Target" for row in rows) == 201
    assert len({(row["sample_id"], row["region"]) for row in rows}) == len(rows)
    assert all(np.isfinite([row["before"], row["after"]]).all() for row in rows)
    provenance = dict(dataset=DOMAIN, support_ids=engine.support_ids, example_id=example,
                      checkpoint=str(engine.path.relative_to(prior.ROOT)), checkpoint_sha256=engine.sha,
                      dataset_manifest_sha256=manifest["manifest_sha256"], queries=n,
                      working_image=SIZE, sam2_input=1024, descriptor_grid=GRID,
                      descriptor_channels=int(keys.shape[1]), background_tile_cells=TILE,
                      target_sampling="Largest GT component, expanded and pooled with mean-max.",
                      background_sampling="Tile with most eligible cells per type, raster-order ties. "
                                          "Exclude cells used for any expanded target. Mean-pool eligible cells. "
                                          "Skip a type when no eligible cells exist.",
                      example_selection="First test-list image with a valid target and distinct tiles for both background types.",
                      comparison="Held-out region descriptors compared with the original and corrected "
                                 "global support target prototypes, before memory content projection.",
                      projection=info)
    return rows, provenance


def region_values(rows, name, key):
    return np.array([float(row[key]) for row in rows if row["region"] == name])


def plot_similarity(rows, provenance):
    manifest = prior.bg.manifest(DOMAIN)
    sid = provenance["example_id"]
    image, mask = prior.base.raw_pair(manifest, sid, SIZE, True)
    image = image.astype(np.uint8)
    regions = query_regions(image, mask > 0)
    fig = plt.figure(figsize=(8.2, 5.25), layout="constrained")
    sections = fig.add_gridspec(2, 1, height_ratios=(1.05, 1.5))
    examples = sections[0].subgridspec(1, 4, wspace=.04)
    overview = fig.add_subplot(examples[0])
    overview.imshow(image)
    overview.set_title(f"IRSTD-1k · {sid}", fontsize=10)
    overview.set_xlabel("Original → corrected")
    overview.set_xticks([])
    overview.set_yticks([])
    overview.spines[:].set_visible(False)
    example_scores = {row["region"]: row for row in rows if row["sample_id"] == sid}
    for i, (name, color, (_, box, pixels)) in enumerate(zip(REGIONS, COLORS, regions), 1):
        x, y, w, h = box
        overview.add_patch(Rectangle((x, y), w, h, fill=False, lw=1.8, ec=color))
        ax = fig.add_subplot(examples[i])
        crop = image[y:y+h, x:x+w]
        alpha = pixels[y:y+h, x:x+w] * .47
        overlay = crop / 255.0 * (1 - alpha[..., None]) + np.array(to_rgb(color)) * alpha[..., None]
        ax.imshow(overlay, interpolation="nearest")
        ax.set_title(name, color=color)
        values = example_scores[name]
        ax.set_xlabel(f"{float(values['before']):.3f} → {float(values['after']):.3f}")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(color)
            spine.set_linewidth(1.6)

    distributions = sections[1].subgridspec(1, 2, wspace=.09)
    summary = []
    vals = [float(row[key]) for row in rows for key in ("before", "after")]
    lower = max(-1., np.floor((min(vals) - .025) * 10) / 10)
    bins = np.linspace(lower, 1., round((1. - lower) / .025) + 1)
    first = fig.add_subplot(distributions[0])
    second = fig.add_subplot(distributions[1], sharey=first)
    for ax, key, title in ((first, "before", "Original support prototype"),
                           (second, "after", "Corrected support prototype")):
        for name, color in zip(REGIONS, COLORS):
            values = region_values(rows, name, key)
            counts, _ = np.histogram(values, bins)
            pmf = counts / len(values)
            ax.stairs(pmf, bins, fill=True, color=color, alpha=.16)
            ax.stairs(pmf, bins, color=color, lw=1.5, label=f"{name} ({len(values)})")
            summary.append(dict(prototype=key, region=name, n=len(values), mean=float(values.mean()),
                                median=float(np.median(values)), q1=float(np.quantile(values, .25)),
                                q3=float(np.quantile(values, .75))))
        ax.set(xlabel="Cosine similarity", title=title, xlim=(lower, 1))
        ax.grid(axis="y", alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
    first.set_ylabel("Query regions per bin")
    first.legend(frameon=False, loc="upper left", fontsize=8)
    save_figure(fig, "fig1b_prototype_similarity")
    return summary


def paired_summary(rows):
    result = []
    for key in ("before", "after"):
        target = {row["sample_id"]: float(row[key]) for row in rows if row["region"] == "Target"}
        for region in REGIONS[1:]:
            background = np.array([float(r[key]) for r in rows if r["region"] == region])
            paired_target = np.array([target[r["sample_id"]] for r in rows if r["region"] == region])
            gap = paired_target - background
            result.append(dict(prototype=key, region=region, pairs=len(gap), mean_gap=float(gap.mean()),
                               background_at_or_above_paired_target=int((gap <= 0).sum()),
                               background_at_or_above_min_target=int((background >= min(target.values())).sum()),
                               background_max=float(background.max()), target_min=min(target.values())))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reuse-similarity", action="store_true")
    args = parser.parse_args()
    plt.rcParams.update({"pdf.fonttype": 42, "font.size": 10, "axes.titleweight": "bold"})
    OUT.mkdir(parents=True, exist_ok=True)
    transfer_and_adaptation()
    csv_path = OUT / "fig1b_heldout_cosines.csv"
    provenance_path = OUT / "fig1b_provenance.json"
    if args.reuse_similarity:
        rows, provenance = prior.csv_rows(csv_path), prior.read_json(provenance_path)
    else:
        rows, provenance = measure()
        prior.save_csv(csv_path, rows)
        prior.bg.dump(provenance_path, provenance)
    summary = plot_similarity(rows, provenance)
    prior.save_csv(OUT / "fig1b_summary.csv", summary)
    prior.save_csv(OUT / "fig1b_paired_gaps.csv", paired_summary(rows))


if __name__ == "__main__":
    main()
