"""File-only teaching versions. Selecting/editing never accesses hardware."""
from copy import deepcopy
from datetime import datetime, timezone
import json
import re
from pathlib import Path
from uuid import uuid4

POINT_NAMES = {'p1': 'p1_standby', 'p2': 'p2_grasp', 'p3': 'p3_lift', 'p4': 'p4_release'}


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temp.replace(path)


def validate_points(points):
    for key, values in points.items():
        if not isinstance(values, list) or len(values) != 6 or any(
                type(v) is not int or not 0 <= v <= 4095 for v in values):
            raise ValueError(f'点位 {key} 必须是 6 个有效舵机刻度')


def derive_waypoints(points, opening=2850):
    """Keep taught joint values intact; rebuild only auxiliary transit targets."""
    wp = {k: deepcopy(points[k]) for k in POINT_NAMES.values() if k in points}
    p1, p2, p3, p4 = (wp.get(k) for k in POINT_NAMES.values())
    if p1 and p2:
        wp['p2_approach'] = [p2[0], int(p1[1]*.5+p2[1]*.5), int(p1[2]*.5+p2[2]*.5), p2[3], p2[4], p1[5]]
        wp['p2_open'] = [p2[0], int(p1[1]*.2+p2[1]*.8), int(p1[2]*.2+p2[2]*.8), p2[3], p2[4], int(opening)]
    if p1 and p3:
        wp['p1_transit'] = [p1[0], int(p3[1]*.6+p1[1]*.4), int(p3[2]*.6+p1[2]*.4), p1[3], p1[4], p1[5]]
    if p2 and p4:
        wp['p4_box'] = p4[:5] + [p2[5]]
    if p3 and p4:
        wp['p4_exit'] = [p4[0], p3[1], p3[2], p4[3], p4[4], p4[5]]
    return wp


def merge_taught_points(existing, updates, opening=2850):
    """Update selected primary points without rewriting unrelated saved transits."""
    if not updates or any(k not in POINT_NAMES.values() for k in updates):
        raise ValueError('请选择 1～4 号示教点位')
    validate_points(updates)
    result = deepcopy(existing)
    changed = {key for key, value in updates.items() if result.get(key) != value}
    result.update(deepcopy(updates))
    if not changed:
        return result
    derived = derive_waypoints(result, opening)
    dependencies = {
        'p2_approach': {'p1_standby', 'p2_grasp'},
        'p2_open': {'p1_standby', 'p2_grasp'},
        'p1_transit': {'p1_standby', 'p3_lift'},
        'p4_box': {'p2_grasp', 'p4_release'},
        'p4_exit': {'p3_lift', 'p4_release'},
    }
    grip_source = {'p2_approach': 'p1_standby', 'p1_transit': 'p1_standby', 'p4_exit': 'p4_release'}
    for key, needed in dependencies.items():
        if key in derived and (key not in existing or needed & changed):
            value = derived[key]
            source = grip_source.get(key)
            if source and key in existing and source in existing and existing[source][5] == result[source][5]:
                value[5] = existing[key][5]
            result[key] = value
    return result


class TeachingVersions:
    def __init__(self, root):
        self.root = Path(root)
        self.path = self.root / 'config/teaching_versions/store.json'
        self.data_root = self.root / 'datasets/studio_versions'
        if self.path.exists():
            self.doc = json.loads(self.path.read_text())
            self.active()  # Reject corrupt selection rather than silently replacing it.
        else:
            legacy = self.root / 'config/auto_waypoints.json'
            backup = self.root / 'config/teaching_versions/legacy_waypoints.json'
            if legacy.exists() and not backup.exists():
                atomic_json(backup, json.loads(legacy.read_text()))
            self.doc = {'schema_version': 1, 'active': None, 'versions': []}
            for name in ('版本1', '版本2', '版本3'):
                self.create(name)
            self.select(self.doc['versions'][0]['id'])

    def _save(self):
        atomic_json(self.path, self.doc)

    def active(self):
        return self.get(self.doc['active'])

    def get(self, ident):
        for v in self.doc['versions']:
            if v['id'] == ident:
                return deepcopy(v)
        raise ValueError('示教版本不存在')

    def directory(self, version=None):
        v = version or self.active()
        # IDs are generated internally; do not use user labels as paths.
        if len(v['id']) != 12 or any(c not in '0123456789abcdef' for c in v['id']):
            raise ValueError('非法版本 ID')
        return self.data_root / v['id'] / f"revision_{int(v['revision']):04d}"

    def episode_files(self, version=None):
        return sorted(self.directory(version).glob('episode_*.json'))

    def summaries(self):
        result = []
        for version in self.doc['versions']:
            v = deepcopy(version)
            v['episodes'] = len(self.episode_files(v))
            v['history_episodes'] = len(list((self.data_root / v['id']).glob('revision_*/episode_*.json'))) - v['episodes']
            v['complete'] = all(k in v['waypoints'] for k in POINT_NAMES.values())
            result.append(v)
        return result

    def create(self, name='', waypoints=None, open_gripper=2850):
        name = name.strip()
        if not name:
            numbers = [int(m.group(1)) for v in self.doc['versions']
                       if (m := re.match(r'^版本(\d+)(?:$|\s|·)', v['name']))]
            name = f'版本{max(numbers, default=0)+1}'
        if len(name) > 60:
            raise ValueError('版本名称需要 1～60 个字符')
        if any(v['name'] == name for v in self.doc['versions']):
            raise ValueError('已有同名版本，请换个名称')
        points = deepcopy(waypoints or {})
        validate_points(points)
        version = {'id': uuid4().hex[:12], 'name': name, 'revision': 1,
                   'waypoints': points, 'open_gripper': int(open_gripper),
                   'updated_at': datetime.now(timezone.utc).isoformat()}
        self.doc['versions'].append(version)
        self.doc['active'] = version['id']
        self._save()
        return deepcopy(version)

    def select(self, ident):
        version = self.get(ident)
        self.doc['active'] = ident
        self._save()
        return version

    def update(self, points, open_gripper=None):
        validate_points(points)
        v = self.active()
        opening = v['open_gripper'] if open_gripper is None else int(open_gripper)
        if not 0 <= opening <= 4095:
            raise ValueError('张爪目标需要在 0～4095 内')
        if v['waypoints'] == points and v['open_gripper'] == opening:
            return v
        # Editing a version that already has recordings creates a new take/revision.
        # Old episodes retain their exact waypoint snapshot and are not overwritten.
        if self.episode_files(v):
            v['revision'] += 1
        v.update(waypoints=deepcopy(points), open_gripper=opening,
                 updated_at=datetime.now(timezone.utc).isoformat())
        self.doc['versions'] = [v if x['id'] == v['id'] else x for x in self.doc['versions']]
        self._save()
        return deepcopy(v)

    def set_points(self, updates):
        v = self.active()
        return self.update(merge_taught_points(v['waypoints'], updates, v['open_gripper']))

    def copy_points(self, source_id, point_keys):
        if not point_keys or any(key not in POINT_NAMES for key in point_keys):
            raise ValueError('请勾选要复制的 1～4 号点位')
        source = self.get(source_id)
        updates = {}
        for key in dict.fromkeys(point_keys):
            name = POINT_NAMES[key]
            if name not in source['waypoints']:
                raise ValueError(f"{source['name']} 尚未保存 {key[1:]} 号点位")
            updates[name] = source['waypoints'][name]
        return self.set_points(updates)

    def training_files(self, scope='current'):
        if scope == 'current':
            versions = [self.active()]
        elif scope == 'all':
            versions = self.summaries()
        else:
            raise ValueError('训练范围必须是 current 或 all')
        return [(v, f) for v in versions for f in self.episode_files(v)]
