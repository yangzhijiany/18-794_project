# Plot early-recognition results and select video examples
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.io import loadmat

from data_process.sampling import sample_frame
from .train_fusion import RATIOS, compare, names, write_csv, write_json

LABELS = {'rgb': 'RGB', 'pose': 'Pose', 'matched': 'Feature fusion',
          'shuffled': 'Shuffled pose'}
COLORS = {'rgb': '#3569a8', 'pose': '#d08426', 'matched': '#238266',
          'shuffled': '#929292'}


def read_csv(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def read_predictions(path, class_names):
    rows = read_csv(path)
    keys = [(r['sample_id'], round(float(r['ratio']), 1)) for r in rows]
    ids = Path('splits/test.txt').read_text(encoding='utf-8-sig').split()
    expected = {(sid, ratio) for sid in ids for ratio in RATIOS}
    if len(set(keys)) != len(keys) or set(keys) != expected:
        raise ValueError(f'{path}: duplicate, missing or extra predictions')
    probs = np.array([[float(r[f'score_{name}']) for name in class_names] for r in rows])
    if not np.isfinite(probs).all() or (probs < 0).any() or (probs > 1).any():
        raise ValueError(f'{path}: invalid probabilities')
    if not np.allclose(probs.sum(1), 1, atol=1e-5):
        raise ValueError(f'{path}: scores must sum to one')
    labels = np.array([int(r['label']) for r in rows])
    predicted = np.array([int(r['pred']) for r in rows])
    if (labels < 0).any() or (labels >= len(class_names)).any():
        raise ValueError(f'{path}: invalid labels')
    if not np.array_equal(predicted, probs.argmax(1)):
        raise ValueError(f'{path}: predictions do not match scores')
    return dict(zip(keys, rows))


def area(x, y):
    # Trapezoidal integration over the measured ratios, without extrapolation
    return sum((x[i + 1] - x[i]) * (y[i] + y[i + 1]) / 2
               for i in range(len(x) - 1))


def curve_and_auc(output):
    rows = read_csv(output / 'accuracy_by_ratio.csv')
    x = np.array(RATIOS)
    original = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
    summary = []
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for row in rows:
        model = row['model']
        y = np.array([float(row[str(r)]) for r in RATIOS])
        original_y = np.array([float(row[str(r)]) for r in original])
        full_auc = area(x, y)
        original_auc = area(original, original_y)
        summary.append({
            'model': model, 'mean_accuracy_7_ratios': float(y.mean()),
            'auc_10_100': full_auc, 'normalized_auc_10_100': full_auc / 0.9,
            'mean_accuracy_5_ratios': float(original_y.mean()),
            'auc_20_100_proposal': original_auc,
            'normalized_auc_20_100_proposal': original_auc / 0.8,
        })
        ax.plot(x * 100, y * 100, label=LABELS[model], color=COLORS[model],
                marker='o', markersize=4, linestyle='--' if model == 'shuffled' else '-')
    ax.set(xlabel='Observed prefix (%)', ylabel='Top-1 accuracy (%)',
           title='Penn Action: accuracy versus observation ratio', xticks=x * 100,
           ylim=(75, 100), xlim=(8, 102))
    ax.grid(alpha=0.2)
    ax.legend(loc='lower right', fontsize=9)
    fig.tight_layout()
    fig.savefig(output / 'accuracy_curve.png', dpi=200)
    fig.savefig(output / 'accuracy_curve.pdf')
    plt.close(fig)
    write_csv(output / 'auc.csv', summary)
    return summary


def confusion_plots(output, predictions, class_names):
    selected = ['rgb', 'pose', 'matched', 'shuffled']
    displayed = [name.replace('_', ' ') for name in class_names]
    mistakes = []
    for ratio in [0.1, 0.2]:
        fig, axes = plt.subplots(2, 2, figsize=(18, 16), layout='constrained')
        for ax, model in zip(axes.flat, selected):
            matrix = np.zeros((len(class_names), len(class_names)), dtype=int)
            for (sample_id, r), row in predictions[model].items():
                if r == ratio:
                    matrix[int(row['label']), int(row['pred'])] += 1
            normalized = matrix / matrix.sum(1, keepdims=True)
            image = ax.imshow(normalized, cmap='Blues', vmin=0, vmax=1)
            for i, j in zip(*np.nonzero(matrix)):
                ax.text(j, i, str(matrix[i, j]), ha='center', va='center', fontsize=8,
                        color='white' if normalized[i, j] > 0.5 else '#263445')
            ax.set(xticks=range(len(class_names)), yticks=range(len(class_names)),
                   xticklabels=displayed, yticklabels=displayed,
                   xlabel='Predicted class', ylabel='True class', title=LABELS[model])
            ax.tick_params(axis='both', labelsize=8)
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            records = []
            for i, action in enumerate(class_names):
                records.append({'true_class': action,
                                **{name: int(matrix[i, j]) for j, name in enumerate(class_names)}})
            write_csv(output / f'confusion_{model}_{int(ratio * 100)}.csv', records)
            errors = [(int(matrix[i, j]), i, j) for i in range(len(class_names))
                      for j in range(len(class_names)) if i != j and matrix[i, j] > 0]
            for count, i, j in sorted(errors, reverse=True)[:10]:
                mistakes.append({'model': model, 'ratio': ratio,
                                 'true_class': class_names[i], 'predicted_class': class_names[j],
                                 'count': count, 'fraction_of_true_class': count / matrix[i].sum()})
        fig.suptitle(f'First {int(ratio * 100)}% of each video | cell labels are sample counts', fontsize=16)
        fig.colorbar(image, ax=axes, shrink=0.65, label='Fraction of true class')
        fig.savefig(output / f'confusion_{int(ratio * 100)}.png', dpi=170)
        fig.savefig(output / f'confusion_{int(ratio * 100)}.pdf')
        plt.close(fig)
    write_csv(output / 'top_confusions.csv', mistakes)


def class_plot(output, class_names):
    rows = read_csv(output / 'per_class.csv')
    early = {row['action']: row for row in rows if row['ratio'] == '0.2'}
    fig, ax = plt.subplots(figsize=(12, 7))
    y = np.arange(len(class_names))
    models = ['rgb', 'pose', 'matched']
    for i, model in enumerate(models):
        values = [float(early[name][model]) * 100 for name in class_names]
        ax.barh(y + (i - 1) * 0.24, values, height=0.23,
                label=LABELS[model], color=COLORS[model])
    ax.set(yticks=y, yticklabels=[n.replace('_', ' ') for n in class_names],
           xlabel='Top-1 accuracy (%)', xlim=(0, 103),
           title='Per-class accuracy after observing the first 20%')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.2)
    ax.set_axisbelow(True)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize=9)
    fig.tight_layout()
    fig.savefig(output / 'per_class_20.png', dpi=200)
    fig.savefig(output / 'per_class_20.pdf')
    plt.close(fig)


def examples(output, data_root, class_names):
    rows = read_csv(output / 'cases.csv')
    chosen, used = [], set()
    # Choose examples at low ratios by video ID, without ranking by confidence
    for category, count in [('improved', 2), ('failed', 1), ('regressed', 1)]:
        candidates = sorted([r for r in rows if r['case'] == category and float(r['ratio']) <= 0.2],
                            key=lambda r: (float(r['ratio']), r['sample_id']))
        labels = set()
        for row in candidates:
            if row['sample_id'] not in used and row['label'] not in labels:
                chosen.append(row.copy())
                used.add(row['sample_id'])
                labels.add(row['label'])
                if len(labels) == count:
                    break
    if len(chosen) != 4:
        raise ValueError('Not enough early examples for all case types')
    if not (data_root / 'frames').is_dir() or not (data_root / 'labels').is_dir():
        write_csv(output / 'case_examples.csv', chosen)
        return False
    fig, axes = plt.subplots(len(chosen), 3, figsize=(12, 12))
    for i, row in enumerate(chosen):
        sid = row['sample_id']
        annotation = loadmat(data_root / 'labels' / f'{sid}.mat')
        action = str(annotation['action'].item()).strip()
        if action != row['label']:
            raise ValueError(f'{sid}: raw annotation and prediction label differ')
        frames = sorted((data_root / 'frames' / sid).glob('*.jpg'), key=lambda p: int(p.stem))
        if len(frames) != annotation['x'].shape[0]:
            raise ValueError(f'{sid}: frame count differs from annotations')
        prefix, sampled = sample_frame(len(frames), float(row['ratio']), 16)
        shown = sampled[[0, 7, 15]].tolist()
        assert max(shown) < prefix
        row['total_frames'] = len(frames)
        row['prefix_frames'] = prefix
        row['shown_frames_1based'] = ','.join(str(j + 1) for j in shown)
        for j, frame_index in enumerate(shown):
            with Image.open(frames[frame_index]) as image:
                axes[i, j].imshow(image.convert('RGB'))
            axes[i, j].set_title(f'Frame {frame_index + 1} / {len(frames)}', fontsize=9)
            axes[i, j].axis('off')
        text = (f"{row['case']} | video {sid} | first {float(row['ratio']):.0%} | GT: {row['label']}\n"
                f"RGB: {row['rgb']} | Pose: {row['pose']} | Fusion: {row['matched']}")
        axes[i, 0].text(0, 1.18, text, transform=axes[i, 0].transAxes,
                        fontsize=10, va='bottom', clip_on=False)
    fig.suptitle('Early recognition examples: frames are restricted to the observed prefix', fontsize=13)
    fig.subplots_adjust(left=0.03, right=0.99, bottom=0.02, top=0.91, hspace=0.8, wspace=0.08)
    fig.savefig(output / 'case_examples.png', dpi=180)
    fig.savefig(output / 'case_examples.pdf')
    plt.close(fig)
    write_csv(output / 'case_examples.csv', chosen)
    return True


def main():
    parser = argparse.ArgumentParser(description='Plot fusion evaluation results')
    parser.add_argument('--rgb', default='runs/rgb_baseline/predictions_test.csv')
    parser.add_argument('--pose', default='runs/pose/bbox_s2/predictions_test.csv')
    parser.add_argument('--matched', default='runs/fusion/matched_s0_rerun/predictions_test.csv')
    parser.add_argument('--shuffled', default='runs/fusion/shuffled_s0_rerun/predictions_test.csv')
    parser.add_argument('--output', default='runs/fusion_analysis_rerun')
    parser.add_argument('--data-root', default='data/Penn_Action')
    args = parser.parse_args()
    compare(args)
    output = Path(args.output)
    class_names = names()
    predictions = {}
    for model in LABELS:
        path = getattr(args, model)
        predictions[model] = read_predictions(path, class_names)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})
    summary = curve_and_auc(output)
    confusion_plots(output, predictions, class_names)
    class_plot(output, class_names)
    has_frames = examples(output, Path(args.data_root), class_names)
    protocol = {
        'ratios': list(RATIOS), 'proposal_ratios': [0.2, 0.4, 0.6, 0.8, 1.0],
        'auc_method': 'trapezoidal, no extrapolation outside measured ratios',
        'auc_units': 'accuracy and observation ratio both on scale 0 to 1',
        'normalization': 'divide AUC by observed interval length: 0.9 or 0.8',
        'confusion_ratios': [0.1, 0.2], 'confusion_rows': 'true class',
        'confusion_columns': 'predicted class', 'matrix_csv_values': 'sample counts',
        'case_selection': 'two improved, one failed, one regressed; first 10% or 20%; video ID order',
        'case_frames_available': has_frames, 'pose': 'ground-truth joints',
        'fusion_training_seed': 0,
    }
    write_json(output / 'evaluation_protocol.json', protocol)
    print('Saved curves, AUC, confusion matrices and case examples to', output)
    for row in summary:
        print(LABELS[row['model']], f"normalized AUC (10-100%): {row['normalized_auc_10_100']:.4f}")


if __name__ == '__main__':
    main()
