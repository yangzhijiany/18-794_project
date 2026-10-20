# Train fusion on cached features from the frozen RGB and pose models
import argparse
import csv
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from .fusion_model import FusionModel

RATIOS = (0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0)


def load(path):
    return torch.load(path, map_location='cpu', weights_only=True)


def names():
    with open('splits/classes.csv', newline='', encoding='utf-8-sig') as f:
        rows = sorted(csv.DictReader(f), key=lambda x: int(x['label_id']))
    if [int(r['label_id']) for r in rows] != list(range(len(rows))):
        raise ValueError('Class IDs must be contiguous from zero')
    return [r['action'] for r in rows]


def checked_cache(path, split, dim):
    data = load(path)
    n = len(data['sample_id'])
    if data['features'].shape != (n, dim):
        raise ValueError(f'{path}: expected features [{n}, {dim}]')
    if data['ratio'].shape != (n,) or data['label'].shape != (n,):
        raise ValueError(f'{path}: invalid metadata shapes')
    if not torch.isfinite(data['features']).all():
        raise ValueError(f'{path}: non-finite features')
    labels = data['label']
    if labels.dtype != torch.long or not ((labels >= 0) & (labels < len(names()))).all():
        raise ValueError(f'{path}: invalid labels')
    ids = Path(f'splits/{split}.txt').read_text(encoding='utf-8-sig').split()
    keys = [(sid, round(float(r), 1)) for sid, r in zip(data['sample_id'], data['ratio'])]
    if not all(isinstance(s, str) for s in data['sample_id']):
        raise ValueError('sample_id must be strings, preserving leading zeros')
    if not all(abs(float(r) - round(float(r), 1)) < 1e-5 for r in data['ratio']):
        raise ValueError('Invalid observation ratios')
    expected = {(s, r) for s in ids for r in RATIOS}
    if len(set(keys)) != n or set(keys) != expected:
        raise ValueError(f'{path}: duplicate/missing/extra (sample_id, ratio) keys')
    for sid in ids:
        if len({int(labels[i]) for i, k in enumerate(keys) if k[0] == sid}) != 1:
            raise ValueError(f'{path}: labels differ across ratios for {sid}')
    return data, {k: i for i, k in enumerate(keys)}


def paired(rgb_dir, pose_dir, split):
    rgb, rgb_index = checked_cache(Path(rgb_dir) / f'features_{split}.pt', split, 2048)
    pose, pose_index = checked_cache(Path(pose_dir) / f'features_{split}.pt', split, 256)
    # Match by video and ratio; the two files may have different row orders
    keys = sorted(rgb_index)
    rgb_rows = torch.tensor([rgb_index[k] for k in keys])
    pose_rows = torch.tensor([pose_index[k] for k in keys])
    if not torch.equal(rgb['label'][rgb_rows], pose['label'][pose_rows]):
        raise ValueError(f'{split}: RGB/Pose labels differ')
    return {
        'ids': [k[0] for k in keys],
        'ratio': torch.tensor([k[1] for k in keys]),
        'label': rgb['label'][rgb_rows],
        'rgb': rgb['features'][rgb_rows].float(),
        'pose': pose['features'][pose_rows].float(),
        'rgb_logits': rgb['logits'][rgb_rows].float() if 'logits' in rgb else None,
        'provenance': {'rgb': rgb.get('checkpoint'), 'pose': pose.get('checkpoint')},
    }


def shuffle_pose(data, seed):
    ids = sorted(set(data['ids']))
    if len(ids) < 2:
        raise ValueError('Shuffled experiment needs at least two videos per split')
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(ids), generator=generator).tolist()
    # Shift the shuffled videos by one so none keeps its own pose
    donor = {}
    for i in range(len(ids)):
        donor[ids[order[i]]] = ids[order[(i + 1) % len(ids)]]
    lookup = {(s, round(float(r), 1)): i for i, (s, r) in enumerate(zip(data['ids'], data['ratio']))}
    # Keep the observation ratio and use the same donor at all seven ratios
    rows = [lookup[(donor[s], round(float(r), 1))] for s, r in zip(data['ids'], data['ratio'])]
    return {**data, 'pose': data['pose'][rows]}, donor


def metrics(logits, data):
    correct = logits.argmax(1).eq(data['label'])
    by_ratio = {}
    for ratio in RATIOS:
        mask = torch.isclose(data['ratio'], torch.tensor(ratio))
        by_ratio[str(ratio)] = correct[mask].float().mean().item()
    return {'mean_accuracy': sum(by_ratio.values()) / len(RATIOS),
            'accuracy_by_ratio': by_ratio}


@torch.no_grad()
def predict(model, data, device, batch_size=512):
    model.eval()
    predictions = []
    for i in range(0, len(data['ids']), batch_size):
        rgb = data['rgb'][i:i + batch_size].to(device)
        pose = data['pose'][i:i + batch_size].to(device)
        predictions.append(model(rgb, pose).cpu())
    return torch.cat(predictions)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2), encoding='utf-8')


def write_csv(path, rows):
    with Path(path).open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_predictions(path, logits, data):
    probs = logits.softmax(1)
    pred = logits.argmax(1)
    class_names = names()
    rows = []
    for i, sample_id in enumerate(data['ids']):
        row = {'sample_id': sample_id, 'ratio': round(float(data['ratio'][i]), 1),
               'label': int(data['label'][i]), 'pred': int(pred[i])}
        row.update({f'score_{name}': float(probs[i, j]) for j, name in enumerate(class_names)})
        rows.append(row)
    write_csv(path, rows)


def train(args):
    out = Path(args.output)
    if out.exists():
        raise FileExistsError(f'Experiment already exists: {out}')
    train_set = paired(args.rgb_dir, args.pose_dir, 'train')
    val_set = paired(args.rgb_dir, args.pose_dir, 'val')
    out.mkdir(parents=True)
    config = vars(args).copy()
    config['provenance'] = {s: d['provenance'] for s, d in [('train', train_set), ('val', val_set)]}
    write_json(out / 'config.json', config)
    if args.mode == 'shuffled':
        train_set, train_map = shuffle_pose(train_set, args.shuffle_seed)
        val_set, val_map = shuffle_pose(val_set, args.shuffle_seed + 1)
        write_json(out / 'pose_donors.json', dict(train=train_map, val=val_map))
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    model = FusionModel(hidden_dim=args.hidden_dim, num_classes=len(names())).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # Training samples one ratio per video; validation checks all seven
    n_videos = len(set(train_set['ids']))
    groups = torch.arange(len(train_set['ids'])).reshape(n_videos, len(RATIOS))
    generator = torch.Generator().manual_seed(args.seed)
    best = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        ratio_indices = torch.randint(len(RATIOS), (n_videos,), generator=generator)
        chosen = groups[torch.arange(n_videos), ratio_indices]
        chosen = chosen[torch.randperm(n_videos, generator=generator)]
        loss_sum = 0.0
        for ix in chosen.split(args.batch_size):
            optimizer.zero_grad(set_to_none=True)
            rgb = train_set['rgb'][ix].to(args.device)
            pose = train_set['pose'][ix].to(args.device)
            labels = train_set['label'][ix].to(args.device)
            logits = model(rgb, pose)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(ix)
        result = metrics(predict(model, val_set, args.device), val_set)
        if result['mean_accuracy'] > best:
            best = result['mean_accuracy']
            torch.save(dict(model=model.state_dict(), model_args=model.model_args,
                            config=config, epoch=epoch), out / 'best.pt')
            write_json(out / 'val_metrics.json', dict(epoch=epoch, **result))
        train_loss = loss_sum / n_videos
        history.append(dict(epoch=epoch, train_loss=train_loss, **result))
        write_json(out / 'history.json', history)
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(f"Epoch {epoch}/{args.epochs} | train loss {train_loss:.4f} | "
                  f"val accuracy {result['mean_accuracy']:.2%} | best {best:.2%}", flush=True)
    print('Best checkpoint:', out / 'best.pt', flush=True)


def test(args):
    state = load(args.checkpoint)
    config = state['config']
    data = paired(args.rgb_dir, args.pose_dir, args.split)
    if config['mode'] == 'shuffled':
        offset = dict(train=0, val=1, test=2)[args.split]
        data, donor = shuffle_pose(data, config['shuffle_seed'] + offset)
        write_json(Path(args.checkpoint).parent / f'pose_donors_{args.split}.json', donor)
    model = FusionModel(**state['model_args']).to(args.device)
    model.load_state_dict(state['model'])
    logits = predict(model, data, args.device)
    out = Path(args.checkpoint).parent
    result = metrics(logits, data)
    write_json(out / f'{args.split}_metrics.json', result)
    write_predictions(out / f'predictions_{args.split}.csv', logits, data)
    if data['rgb_logits'] is not None:
        write_predictions(out / f'predictions_rgb_{args.split}.csv', data['rgb_logits'], data)
        write_json(out / f'rgb_{args.split}_metrics.json', metrics(data['rgb_logits'], data))
    print(json.dumps(result, indent=2))


def compare(args):
    def read(path):
        with open(path, newline='', encoding='utf-8-sig') as f:
            rows = list(csv.DictReader(f))
        keyed = {(r['sample_id'], round(float(r['ratio']), 1)): r for r in rows}
        if len(keyed) != len(rows):
            raise ValueError(f'Duplicate prediction keys: {path}')
        return keyed
    sources = {'rgb': read(args.rgb), 'pose': read(args.pose),
               'matched': read(args.matched), 'shuffled': read(args.shuffled)}
    keys = set(sources['rgb'])
    if not all(set(d) == keys for d in sources.values()):
        raise ValueError('Prediction files have different sample keys')
    if not all(len({int(d[k]['label']) for d in sources.values()}) == 1 for k in keys):
        raise ValueError('Prediction files have inconsistent labels')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    class_names = names()
    summary = []
    for name, predictions in sources.items():
        row = {'model': name}
        for ratio in RATIOS:
            subset = [predictions[k] for k in sorted(keys) if k[1] == ratio]
            if not subset:
                raise ValueError(f'Missing ratio {ratio}')
            correct = sum(int(x['label']) == int(x['pred']) for x in subset)
            row[str(ratio)] = correct / len(subset)
        row['mean'] = sum(row[str(r)] for r in RATIOS) / len(RATIOS)
        summary.append(row)
    write_csv(out / 'accuracy_by_ratio.csv', summary)
    classes = []
    for c, action in enumerate(class_names):
        for ratio in (*RATIOS, 'all'):
            subset = [k for k in sorted(keys) if int(sources['rgb'][k]['label']) == c
                      and (ratio == 'all' or k[1] == ratio)]
            if not subset:
                continue
            row = dict(action=action, ratio=ratio, count=len(subset))
            for name, predictions in sources.items():
                correct = sum(int(predictions[k]['pred']) == c for k in subset)
                row[name] = correct / len(subset)
            row['gain_vs_rgb'] = row['matched'] - row['rgb']
            row['gain_vs_pose'] = row['matched'] - row['pose']
            classes.append(row)
    write_csv(out / 'per_class.csv', classes)
    cases = []
    for k in sorted(keys):
        label = int(sources['rgb'][k]['label'])
        rgb_ok = int(sources['rgb'][k]['pred']) == label
        fused_ok = int(sources['matched'][k]['pred']) == label
        if fused_ok and not rgb_ok:
            case = 'improved'
        elif rgb_ok and not fused_ok:
            case = 'regressed'
        elif fused_ok:
            case = 'correct'
        else:
            case = 'failed'
        row = {'sample_id': k[0], 'ratio': k[1], 'label': class_names[label], 'case': case}
        for name, predictions in sources.items():
            row[name] = class_names[int(predictions[k]['pred'])]
        cases.append(row)
    write_csv(out / 'cases.csv', cases)
    print(json.dumps(summary, indent=2))


def main():
    p = argparse.ArgumentParser(description='Train and evaluate RGB-pose fusion')
    sub = p.add_subparsers(dest='command', required=True)
    for command in ('check', 'train', 'test'):
        s = sub.add_parser(command)
        s.add_argument('--rgb-dir', default='runs/features_rgb')
        s.add_argument('--pose-dir', default='runs/pose/bbox_s2')
        s.add_argument('--device', default='cpu')
        s.add_argument('--threads', type=int, default=1)
        if command == 'train':
            s.add_argument('--output', required=True)
            s.add_argument('--mode', choices=['matched', 'shuffled'], default='matched')
            s.add_argument('--seed', type=int, default=0)
            s.add_argument('--shuffle-seed', type=int, default=12345)
            s.add_argument('--epochs', type=int, default=100)
            s.add_argument('--batch-size', type=int, default=64)
            s.add_argument('--hidden-dim', type=int, default=256)
            s.add_argument('--lr', type=float, default=0.001)
            s.add_argument('--weight-decay', type=float, default=0.0001)
        if command == 'test':
            s.add_argument('--checkpoint', required=True)
            s.add_argument('--split', choices=['val', 'test'], default='test')
    s = sub.add_parser('compare')
    for name in ('rgb', 'pose', 'matched', 'shuffled'):
        s.add_argument(f'--{name}', required=True)
    s.add_argument('--output', default='runs/fusion_analysis')
    args = p.parse_args()
    if hasattr(args, 'threads'):
        torch.set_num_threads(args.threads)
    if args.command == 'check':
        sets = {s: paired(args.rgb_dir, args.pose_dir, s) for s in ('train', 'val', 'test')}
        for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
            if set(sets[a]['ids']) & set(sets[b]['ids']):
                raise ValueError(f'{a}/{b} splits overlap')
        for s, d in sets.items():
            print(s, len(d['ids']), tuple(d['rgb'].shape), tuple(d['pose'].shape))
    elif args.command == 'train':
        if args.epochs < 1 or args.batch_size < 1:
            p.error('epochs and batch-size must be positive')
        train(args)
    elif args.command == 'test':
        test(args)
    else:
        compare(args)


if __name__ == '__main__':
    main()
