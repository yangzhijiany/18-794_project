# Evaluate the original RGB classifier on cached features
import argparse
from pathlib import Path

import torch
from torch.nn import functional as F

from .train_fusion import checked_cache, load, metrics, write_json, write_predictions


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description='Evaluate cached RGB features')
    p.add_argument('--rgb-dir', required=True)
    p.add_argument('--checkpoint', required=True,
                   help='RGB checkpoint used to extract the features')
    p.add_argument('--split', choices=['val', 'test'], default='test')
    p.add_argument('--output', default='runs/rgb_baseline')
    args = p.parse_args()
    torch.set_num_threads(1)
    cache, _ = checked_cache(Path(args.rgb_dir) / f'features_{args.split}.pt', args.split, 2048)
    state = load(args.checkpoint)['model']
    # Apply the original classifier before any L2 normalization
    logits = F.linear(cache['features'].float(), state['new_fc.weight'], state['new_fc.bias'])
    data = dict(ids=cache['sample_id'], ratio=cache['ratio'], label=cache['label'])
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    write_predictions(out / f'predictions_{args.split}.csv', logits, data)
    result = metrics(logits, data)
    write_json(out / f'{args.split}_metrics.json', result)
    write_json(out / 'provenance.json', dict(feature_checkpoint=cache.get('checkpoint'),
        classifier_checkpoint=str(Path(args.checkpoint).resolve()),
        method='original linear head applied to cached, unnormalized RGB features'))
    print(result)


if __name__ == '__main__':
    main()
