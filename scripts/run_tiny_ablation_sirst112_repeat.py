"""Independent SIRST follow-up; never replace the original ablation result."""
import argparse

import run_tiny_ablation_combined as worker
from run_tiny_ablation_three_seeds import FIELDS, ROOT, read, sha, write

OUT = ROOT / 'outputs/tiny_ablation_other_datasets_20261111'
ORIGINAL = OUT / 'seed112/SIRST/combined'
REPEAT = OUT / 'repeats/sirst_seed112_combined'
METRICS = 'epoch_0120/metrics/summary.json'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--seed', type=int, choices=(112, 114), default=112)
    args = parser.parse_args()
    output = REPEAT if args.seed == 112 else OUT / 'repeats/sirst_seed114_combined'
    config = read(ORIGINAL / 'config.json')
    original = read(ORIGINAL / METRICS)
    assert (config['dataset'], config['seed'], config['variant']) == ('SIRST', 112, 'combined')
    assert original['predictions_recount_verified'] and original['all_120_epochs_verified']
    for path, expected in config['source_hashes'].items():
        assert sha(ROOT / path) == expected, f'Original source differs: {path}'
    worker.OUT = output
    worker.RUN_SEED = args.seed
    path = output / 'combined' / METRICS
    if args.smoke or not path.exists():
        worker.worker(argparse.Namespace(dataset='SIRST', variant='combined', smoke=args.smoke))
    if args.smoke:
        return
    repeated = read(path)
    actual = read(output / 'combined/config.json')
    expected_config = dict(config, seed=args.seed, training_random_seed=args.seed,
                           matched_reference=(output / 'native' / METRICS).relative_to(ROOT).as_posix())
    assert actual == expected_config
    assert repeated['seed'] == args.seed
    assert repeated['predictions_recount_verified'] and repeated['all_120_epochs_verified']
    assert sha(output / 'combined/checkpoints/epoch_0120.pt') == repeated['checkpoint_sha256']
    write(output / 'comparison.json', dict(
        status='complete', dataset='SIRST', seed=args.seed, original_seed=112, variant='combined',
        original_path=(ORIGINAL / METRICS).relative_to(ROOT).as_posix(),
        repeat_path=path.relative_to(ROOT).as_posix(),
        original={k: original[k] for k in FIELDS}, repeat={k: repeated[k] for k in FIELDS},
        delta={k: repeated[k]-original[k] for k in FIELDS},
        original_in_primary_ablation=True))


if __name__ == '__main__':
    main()
