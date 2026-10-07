"""Plot completed epochs from the JSONL training log without smoothing."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, default=Path('outputs/training/mobilesam_kv_r8_mask_output'))
    args = parser.parse_args()
    rows = [json.loads(line) for line in (args.run_dir / 'history.jsonl').read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError('No completed epochs')
    epochs = [r['epoch'] for r in rows]
    if len(set(epochs)) != len(epochs):
        raise ValueError('Duplicate epochs: resolve resumed-run history before plotting')
    rows.sort(key=lambda r: r['epoch'])
    epochs = [r['epoch'] for r in rows]
    font = Path('C:/Windows/Fonts/msyh.ttc')
    if font.exists():
        plt.rcParams['font.family'] = FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({'axes.unicode_minus': False, 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), layout='constrained')
    fig.suptitle('MobileSAM · K/V LoRA + 解冻 mask 输出端', fontsize=18, fontweight='bold')
    panels = [
        ('训练与验证损失 ↓', 'BCE + Dice loss', [('train_loss', '训练', '#2563eb', 1), ('loss', '验证', '#e87924', 1)]),
        ('验证集分割指标 ↑', '%', [('iou', '累计前景 IoU', '#2563eb', 100), ('f1', 'F1', '#059669', 100), ('mean_image_iou', '逐图平均 IoU', '#9333ea', 100)]),
        ('验证集检出率 Pd ↑', '%', [('pd', 'Pd', '#059669', 100)]),
        ('验证集虚警 Fa ↓', '每百万像素', [('fa_per_million', 'Fa', '#dc2626', 1)]),
    ]
    for ax, (title, ylabel, series) in zip(axes.flat, panels):
        for key, label, color, scale in series:
            vals = [r[key] * scale for r in rows]
            ax.plot(epochs, vals, '-o', color=color, label=label, linewidth=2, markersize=4)
            ax.annotate(f'{vals[-1]:.3f}' if scale == 1 and key != 'fa_per_million' else f'{vals[-1]:.2f}',
                        (epochs[-1], vals[-1]), xytext=(6, 0), textcoords='offset points', color=color, fontsize=10)
        ax.set(title=title, xlabel='Epoch', ylabel=ylabel)
        ax.grid(alpha=.18)
        ax.set_xticks(epochs if len(epochs) <= 20 else epochs[::max(1, len(epochs)//10)])
        ax.set_xlim(min(epochs)-.3, max(epochs)+1)
        ax.set_ylim(bottom=0)
        if ylabel == '%':
            ax.set_ylim(0, 100)
        ax.legend(loc='best', frameon=False)
    best = max(rows, key=lambda r: r['iou'])
    fig.supxlabel(f"640 张训练 / {rows[-1]['samples']} 张验证 · 未平滑 · 最佳验证 IoU：Epoch {best['epoch']}，{best['iou']*100:.2f}%", fontsize=11)
    out = args.run_dir / 'figures'
    out.mkdir(exist_ok=True)
    for ext in ('png', 'pdf'):
        path = out / f'training_curves.{ext}'
        fig.savefig(path, dpi=180, facecolor='white')
        print(path.resolve())
    plt.close(fig)


if __name__ == '__main__':
    main()
