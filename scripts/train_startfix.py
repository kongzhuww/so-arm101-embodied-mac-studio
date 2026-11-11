"""Offline-only startup ablation: immutable source, shorter idle, longer ACT chunk.

Default prepares and validates data only. --train starts training, never hardware.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
from datetime import datetime, timezone

from studio_samples import trim_startup, resample_episode

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'datasets/studio_dataset_v2_20261106_170913'
STATUS = ROOT / 'reports/startfix-status.json'


def prepare(source, dest):
    files = sorted(source.glob('episode_*.json'))
    if len(files) != 10:
        raise ValueError(f'Expected 10 source episodes, got {len(files)}')
    planned = []
    for file in files:
        content = file.read_bytes()
        rows = json.loads(content)
        cropped = trim_startup(rows)
        samples = resample_episode(cropped)
        if len(samples) < 50:
            raise ValueError('Episode shorter than proposed 50-frame chunk')
        planned.append((file, content, rows, cropped, samples))
    dest.mkdir(parents=True, exist_ok=False)
    records = []
    for file, content, rows, cropped, samples in planned:
        (dest / file.name).write_text(json.dumps(cropped))
        records.append({'file': file.name, 'source_sha256': hashlib.sha256(content).hexdigest(),
                        'original_frames': len(rows), 'cropped_frames': len(cropped),
                        'resampled_frames': len(samples),
                        'removed_seconds': cropped[0]['timestamp']-rows[0]['timestamp'],
                        'first_measured': cropped[0]['state'],
                        'first_command': cropped[0]['action']})
    manifest = {'source': str(source), 'derived': str(dest), 'episodes': records,
                'standby_shoulder': 1025, 'departure_counts': 75, 'lead_seconds': 0.6,
                'chunk_size': 50, 'n_action_steps': 20, 'total_steps': 3000}
    (dest / 'preparation.json').write_text(json.dumps(manifest, indent=2))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', action='store_true')
    args = parser.parse_args()
    tag = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    dest = ROOT / 'datasets' / f'studio_startfix_{tag}'
    manifest = prepare(SOURCE, dest)
    status = {'pid': os.getpid(), 'status': 'prepared', 'derived': str(dest),
              'chunk_size': 50, 'n_action_steps': 20}
    def save():
        temp = STATUS.with_suffix('.json.tmp')
        temp.write_text(json.dumps(status, indent=2)); temp.replace(STATUS)
    save()
    for r in manifest['episodes']:
        print(f"{r['file']}: removed {r['removed_seconds']:.2f}s; retained {r['resampled_frames']} @10Hz", flush=True)
    if not args.train:
        print('PREPARE ONLY: no training or hardware started')
        return
    try:
        from train_studio_pipeline import run_training_pipeline
        status['status'] = 'training'; save()
        def progress(step, loss):
            status.update(step=step, loss=loss); save()
        final = run_training_pipeline(dest, progress_callback=progress,
                    chunk_size=50, n_action_steps=20, total_steps=3000)
        status.update(status='completed', model=str(final)); save()
        print('No robot inference was started.', flush=True)
    except BaseException as exc:
        status.update(status='failed', error=repr(exc)); save()
        raise


if __name__ == '__main__':
    main()
