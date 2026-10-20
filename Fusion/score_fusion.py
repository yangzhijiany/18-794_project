# Combine RGB and pose probabilities with a weight selected on validation
import argparse
import csv
from pathlib import Path

import torch

from .train_fusion import RATIOS, names, write_csv, write_json


def read_predictions(path, split):
    class_names = names()
    score_columns = [f'score_{name}' for name in class_names]
    with Path(path).open(newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        if not set(['sample_id', 'ratio', 'label', *score_columns]).issubset(reader.fieldnames or []):
            raise ValueError(f'{path}: missing prediction columns')
        rows = list(reader)

    predictions = {}
    for row in rows:
        ratio = float(row['ratio'])
        if abs(ratio - round(ratio, 1)) > 1e-5:
            raise ValueError(f'{path}: invalid ratio')
        key = (row['sample_id'], round(ratio, 1))
        if key in predictions:
            raise ValueError(f'{path}: duplicate prediction {key}')
        label = int(row['label'])
        if not 0 <= label < len(class_names):
            raise ValueError(f'{path}: invalid label')
        predictions[key] = (label, [float(row[c]) for c in score_columns])

    ids = Path(f'splits/{split}.txt').read_text(encoding='utf-8-sig').split()
    expected = {(sample_id, ratio) for sample_id in ids for ratio in RATIOS}
    if set(predictions) != expected:
        raise ValueError(f'{path}: missing or extra video-ratio pairs')
    keys = sorted(predictions)
    labels = torch.tensor([predictions[k][0] for k in keys])
    probs = torch.tensor([predictions[k][1] for k in keys], dtype=torch.float64)
    if not torch.isfinite(probs).all() or (probs < 0).any() or (probs > 1).any():
        raise ValueError(f'{path}: invalid probabilities')
    if not torch.allclose(probs.sum(1), torch.ones(len(keys), dtype=probs.dtype), atol=1e-5):
        raise ValueError(f'{path}: scores must be softmax probabilities')
    return keys, labels, probs


def paired_predictions(rgb_path, pose_path, split):
    keys, labels, rgb = read_predictions(rgb_path, split)
    pose_keys, pose_labels, pose = read_predictions(pose_path, split)
    if keys != pose_keys or not torch.equal(labels, pose_labels):
        raise ValueError(f'{split}: RGB and pose predictions do not match')
    return keys, labels, rgb, pose


def accuracy(probs, keys, labels):
    correct = probs.argmax(1).eq(labels)
    ratios = torch.tensor([k[1] for k in keys], dtype=torch.float64)
    by_ratio = {}
    for ratio in RATIOS:
        mask = torch.isclose(ratios, torch.tensor(ratio, dtype=ratios.dtype))
        by_ratio[str(ratio)] = correct[mask].double().mean().item()
    return {'mean_accuracy': sum(by_ratio.values()) / len(RATIOS),
            'accuracy_by_ratio': by_ratio}


def select_weight(keys, labels, rgb, pose):
    results = []
    for i in range(11):
        alpha = i / 10
        probs = alpha * rgb + (1 - alpha) * pose
        result = accuracy(probs, keys, labels)
        results.append({'rgb_weight': alpha, 'pose_weight': round(1 - alpha, 1), **result})
    # Break validation ties by choosing the weight closest to 0.5, then smaller alpha
    best = max(results, key=lambda r: (round(r['mean_accuracy'], 12),
                                      -abs(r['rgb_weight'] - 0.5), -r['rgb_weight']))
    return best, results


def save_predictions(path, keys, labels, probs):
    class_names = names()
    pred = probs.argmax(1)
    rows = []
    for i, (sample_id, ratio) in enumerate(keys):
        row = {'sample_id': sample_id, 'ratio': ratio,
               'label': int(labels[i]), 'pred': int(pred[i])}
        # These are already probabilities; do not apply softmax again
        for j, name in enumerate(class_names):
            row[f'score_{name}'] = float(probs[i, j])
        rows.append(row)
    write_csv(path, rows)


def main():
    parser = argparse.ArgumentParser(description='Weighted RGB-pose probability fusion')
    parser.add_argument('--rgb-dir', default='runs/rgb_baseline')
    parser.add_argument('--pose-dir', default='runs/pose/bbox_s2')
    parser.add_argument('--output', default='runs/score_fusion')
    args = parser.parse_args()
    torch.set_num_threads(1)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f'Experiment already exists: {output}')

    # Select one weight using validation only, shared by all observation ratios
    keys, labels, rgb, pose = paired_predictions(
        Path(args.rgb_dir) / 'predictions_val.csv',
        Path(args.pose_dir) / 'predictions_val.csv', 'val')
    best, results = select_weight(keys, labels, rgb, pose)
    alpha = best['rgb_weight']
    output.mkdir(parents=True)
    write_json(output / 'validation_search.json', results)
    write_json(output / 'config.json', {
        **vars(args), 'rgb_weight': alpha, 'pose_weight': best['pose_weight'],
        'selection': 'mean validation accuracy over seven ratios',
        'tie_break': 'closest to 0.5, then smaller RGB weight',
    })
    write_json(output / 'val_metrics.json', best)
    save_predictions(output / 'predictions_val.csv', keys, labels,
                     alpha * rgb + (1 - alpha) * pose)
    print(f'Validation selected RGB={alpha:.1f}, pose={1 - alpha:.1f}', flush=True)

    # Apply the selected weight once to test predictions
    keys, labels, rgb, pose = paired_predictions(
        Path(args.rgb_dir) / 'predictions_test.csv',
        Path(args.pose_dir) / 'predictions_test.csv', 'test')
    probs = alpha * rgb + (1 - alpha) * pose
    result = accuracy(probs, keys, labels)
    write_json(output / 'test_metrics.json', result)
    save_predictions(output / 'predictions_test.csv', keys, labels, probs)
    print(f"Test mean accuracy: {result['mean_accuracy']:.2%}")


if __name__ == '__main__':
    main()
