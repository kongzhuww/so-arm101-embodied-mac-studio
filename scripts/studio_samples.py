"""Hardware-independent validation for real studio observation/action records."""
import base64
import cv2
import numpy as np


def decode_sample(row):
    if row.get('schema_version') != 2:
        raise ValueError('旧示范缺少真实图像/从臂读数，不能用于视觉训练；原文件保留')
    for key in ('state', 'action'):
        a = np.asarray(row[key], dtype=float)
        if a.shape != (6,) or not np.isfinite(a).all():
            raise ValueError(f'无效的 {key}')
    if not 0 <= row['timestamp'] - row['image_timestamp'] <= 0.25:
        raise ValueError('摄像头帧过期或时间戳不匹配')
    raw = base64.b64decode(row['image_jpeg'], validate=True)
    bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError('无法解码录制图像')
    return cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (320, 240))


def validate_episode(rows):
    if len(rows) < 11:
        raise ValueError('示范帧数不足')
    last = -float('inf')
    for row in rows:
        decode_sample(row)
        if row['timestamp'] <= last:
            raise ValueError('示范时间戳必须递增')
        last = row['timestamp']


def trim_startup(rows, *, standby_shoulder=1025, departure_counts=75, lead_seconds=0.6):
    """Remove redundant startup only; retain original rows and real timestamps.

    Project-specific onset: shoulder command moves >75 counts past standby.
    Keep 0.6s before onset, rather than shifting future actions onto old images.
    """
    validate_episode(rows)
    onset = next((r['timestamp'] for r in rows
                  if r['action'][1] > standby_shoulder + departure_counts), None)
    if onset is None:
        raise ValueError('未找到离开待机位的动作，拒绝自动裁剪')
    cutoff = onset - lead_seconds
    start = next((i for i, row in enumerate(rows) if row['timestamp'] >= cutoff), 0)
    trimmed = rows[start:]
    validate_episode(trimmed)
    return trimmed


def resample_episode(rows, fps=10):
    """Use actual capture times rather than relabeling 20/30 Hz as 10 Hz."""
    validate_episode(rows)
    times = np.array([r['timestamp'] for r in rows])
    grid = np.arange(times[0], times[-1] + 1e-6, 1 / fps)
    indices = np.searchsorted(times, grid).clip(0, len(rows)-1)
    previous = (indices-1).clip(0)
    indices = np.where(abs(times[previous]-grid) < abs(times[indices]-grid), previous, indices)
    if np.max(abs(times[indices]-grid)) > 0.15:
        raise ValueError('录制存在超过容差的采样空档，请检查摄像头/串口')
    return [rows[int(i)] for i in indices]
