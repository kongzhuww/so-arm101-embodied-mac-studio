"""Restore version1 from historical middle demonstrations. No hardware access.

Original files and model weights are never edited. Current version store is backed up.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

from teaching_versions import TeachingVersions, atomic_json, validate_points
from studio_samples import validate_episode

ROOT = Path(__file__).resolve().parents[1]


def recover(root):
    root = Path(root)
    historical = root/'datasets/studio_dataset_20261106_155432/episode_0000.json'
    segments=[]
    for row in json.loads(historical.read_text()):
        if not segments or row['action'] != segments[-1]:
            segments.append(row['action'])
    if len(segments)!=10 or segments[0]!=segments[-1]:
        raise ValueError('Historical segment layout differs; refuse guessed restoration')
    names=('p1_standby','p2_approach','p2_open','p2_grasp','p3_lift',
           'p4_box','p4_release','p4_exit','p1_transit')
    points=dict(zip(names,segments[:9]))
    validate_points(points)
    source=root/'datasets/studio_dataset_v2_20261106_170913'
    sources=[source/f'episode_{i:04d}.json' for i in range(10)]
    for path in sources:
        data=json.loads(path.read_text())
        validate_episode(data)
        commands=[r['action'] for r in data]
        if points['p2_grasp'] not in commands or points['p4_release'] not in commands:
            raise ValueError(f'{path.name} does not match successful middle endpoints')
    return points,sources,historical


def apply(root=ROOT):
    root=Path(root)
    points,sources,historical=recover(root)
    store=TeachingVersions(root)
    tag=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    backup=store.path.with_name(f'store.before-restore-{tag}.json')
    shutil.copy2(store.path,backup)
    first=store.doc['versions'][0]
    store.select(first['id'])
    # Preserve any existing current-revision samples, even if the points match.
    if store.episode_files():
        for v in store.doc['versions']:
            if v['id']==first['id']:
                v['revision']+=1
    store.update(points,points['p2_open'][5])
    for i,v in enumerate(store.doc['versions'],1):
        v['name']=f'版本{i}'
    store._save()
    version=store.active()
    target=store.directory(version)
    if target.exists() and list(target.glob('episode_*.json')):
        raise ValueError('Target revision is not empty; refuse overwrite')
    target.mkdir(parents=True,exist_ok=True)
    copied=[]
    for source in sources:
        destination=target/source.name
        shutil.copy2(source,destination)
        digest=hashlib.sha256(source.read_bytes()).hexdigest()
        if hashlib.sha256(destination.read_bytes()).hexdigest()!=digest:
            raise ValueError('Restored data checksum mismatch')
        atomic_json(target/'metadata'/source.name,{
            'version':version,'restored_from':str(source),'source_sha256':digest,
            'restored_at':datetime.now(timezone.utc).isoformat()})
        copied.append(str(destination))
    report={'active_version':version,'original_store_backup':str(backup),
            'waypoint_source':str(historical),'restored_episodes':copied,
            'model_unchanged':str(root/'outputs/act_20261106T174035Z/final')}
    atomic_json(root/'reports/restored-version1.json',report)
    return report


if __name__=='__main__':
    report=apply()
    print(json.dumps({'version':report['active_version'],
                      'restored_episodes':len(report['restored_episodes']),
                      'backup':report['original_store_backup']},ensure_ascii=False,indent=2))
