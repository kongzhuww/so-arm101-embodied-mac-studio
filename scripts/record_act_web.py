"""Real kinesthetic demonstrations: read-only motors + camera, local LeRobot data.

No torque/configuration/position writes. Human moves an already torque-off arm.
Action labels are the NEXT OBSERVED joint state (not issued motor commands).
This is hand-guided demonstration data, not phone teleoperation command data.
"""
import argparse
from contextlib import closing
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import threading
import time

import cv2
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
import uvicorn

import read_only_probe as readonly

ROOT = Path(__file__).resolve().parents[1]
HOST, PORT = '10.19.220.84', 4466
ORIGIN = f'http://{HOST}:{PORT}'
NAMES = ['shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll','gripper']
TASK = 'Pick up the spare servo and place it into the open box.'
HTML = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ACT示范录制</title><style>body{font:18px system-ui;background:#111827;color:white;max-width:600px;margin:24px auto;padding:16px}button{width:100%;padding:22px;margin:10px 0;font:inherit;border:0;border-radius:12px;background:#2563eb;color:white}#save{background:#15803d}#train{background:#7c3aed}pre{white-space:pre-wrap}img{width:100%;min-height:60px}p{line-height:1.5;color:#cbd5e1}</style>
<h2>ACT · 手带示范录制</h2><pre id="s">连接中…</pre><img id="view" alt="开始录制后显示相机画面">
<button id="start">① 开始一条</button><button id="save">② 完成任务，保存这一条</button><button id="discard">这条没做好，丢弃</button><button id="train">③ 结束采集，开始 ACT 训练</button>
<p>录制时六轴必须已关闭扭矩。轻轻带着机械臂完成“抓物体→放进盒子”，不要硬掰难推的关节；保持物体、夹爪和盒子在相机画面内。先等状态显示“正在录制”再操作。</p>
<p>先录5条完整成功示范即可启动首版；更多、多样的成功示范通常更有帮助。每条最长120秒。动作标签是下一帧实测关节状态，不是电机指令。</p>
<p>开始训练后可离开，不会自主驱动机械臂。首次训练会下载官方ResNet18图像预训练权重（约45MB）；全部示范和模型保存在Mac本机，不上传。</p>
<script>
const s=document.querySelector('#s');let lastMode='';
async function refresh(){try{const r=await fetch('/status');const d=await r.json();s.textContent=JSON.stringify(d,null,2);
 document.querySelector('#start').disabled=d.mode!=='idle';document.querySelector('#save').disabled=d.mode!=='recording';document.querySelector('#discard').disabled=d.mode!=='recording';document.querySelector('#train').disabled=d.mode!=='idle'||d.episodes<1;
 if(d.mode==='recording')document.querySelector('#view').src='/view?t='+Date.now();lastMode=d.mode;}catch(e){s.textContent='服务连接断开，请看Pi。'}}
async function send(path){try{const r=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'});const d=await r.json();if(!r.ok)alert(d.message);await refresh();}catch(e){alert('请求结果不明，不要重复开始；查看Pi。')}}
document.querySelector('#start').onclick=()=>send('/record');document.querySelector('#save').onclick=()=>send('/save');document.querySelector('#discard').onclick=()=>send('/discard');document.querySelector('#train').onclick=()=>{if(confirm('结束采集并开始本机训练？不会驱动机械臂。'))send('/train')};setInterval(refresh,1000);refresh();
</script></html>'''


def normalize(raw, calibration):
    values = []
    for name, val in zip(NAMES, raw):
        c = calibration[name]
        lo, hi = c['range_min'], c['range_max']
        if name == 'gripper':
            values.append((min(hi, max(lo, val))-lo)*100/(hi-lo))
        else:
            values.append((val-(lo+hi)/2)*360/4095)
    return np.asarray(values, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not args.execute:
        print('PLAN ONLY: real torque-off hand-guided data; no camera, serial or dataset opened.')
        return
    from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig
    from lerobot.datasets import LeRobotDataset
    tag = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    root = ROOT/'datasets'/('so101_pickplace_'+tag)
    output = ROOT/'outputs'/('act_'+tag)
    calibration = json.loads((ROOT/'calibration/so101-follower-20261031T144217-014032Z.json').read_text())
    dataset = LeRobotDataset.create(
        repo_id='local/so101_pickplace', root=root, fps=10,
        robot_type='so101_follower', use_videos=False, image_writer_threads=2,
        features={
            'observation.state': {'dtype':'float32','shape':(6,), 'names':NAMES},
            'action': {'dtype':'float32','shape':(6,), 'names':NAMES},
            'observation.images.external_front': {'dtype':'image','shape':(240,320,3), 'names':['height','width','channels']},
        })
    (root/'recording-method.json').write_text(json.dumps({
        'method':'torque_off_kinesthetic', 'action_labels':'next_observed_joint_state',
        'motor_commands_issued':False, 'fps':10, 'task':TASK,
        'units':'IDs1..5 degrees around recorded midpoint; ID6 percent open',
    }, indent=2))
    state = {'mode':'idle','episodes':0,'frames_this_episode':0,'dataset':str(root),
             'message':'准备好物体和盒子后，点开始一条。不会驱动机械臂。'}
    stop_record = threading.Event()
    discard = threading.Event()
    lock = threading.Lock()
    image = {'jpeg':None}
    trainer = {'process':None}
    app = FastAPI()

    @app.middleware('http')
    async def restrict(request: Request, call_next):
        if request.client.host not in {'10.19.220.45', HOST, '127.0.0.1'}:
            return JSONResponse({'message':'只允许手机热点设备'}, status_code=403)
        if request.method == 'POST' and request.headers.get('origin') != ORIGIN:
            return JSONResponse({'message':'禁止跨站请求'}, status_code=403)
        return await call_next(request)

    @app.get('/', response_class=HTMLResponse)
    def index(): return HTMLResponse(HTML, headers={'Cache-Control':'no-store'})

    @app.get('/status')
    def status():
        process = trainer['process']
        if process is not None and process.poll() is not None:
            state['mode'] = 'trained' if process.returncode == 0 else 'training_error'
            state['message'] = '训练完成，无自主运动。' if process.returncode == 0 else '训练出错，请看Pi日志；不会驱动机械臂。'
        progress = output/'training-status.json'
        result = dict(state)
        if progress.exists():
            try: result['training'] = json.loads(progress.read_text())
            except (ValueError, OSError): pass
        return result

    @app.get('/view')
    def view():
        data = image['jpeg']
        return Response(data or b'', media_type='image/jpeg', status_code=200 if data else 204)

    def capture():
        camera = None
        try:
            camera = OpenCVCamera(OpenCVCameraConfig(index_or_path=0, fps=60, width=640, height=480, backend=1200))
            camera.connect()
            with closing(readonly.open_verified_port()) as port:
                link = readonly.ReadOnlyLink(port)
                readonly.probe(link, emit=lambda _:None)
                state['mode'] = 'recording'
                state['message'] = '正在录制：轻轻带着机械臂完成任务，完成后点保存。'
                previous = None
                count = 0
                started = time.monotonic()
                timing = []
                while not stop_record.is_set() and time.monotonic()-started < 120:
                    tick = time.monotonic()
                    for i in readonly.IDS:
                        if link.read_register(i, 'torque_enable') != 0:
                            raise RuntimeError('采集需要六轴扭矩关闭；发现非零，停止只读采集')
                    raw = [link.read_register(i, 'present_position') for i in readonly.IDS]
                    joints = normalize(raw, calibration)
                    rgb = camera.async_read(timeout_ms=500)
                    if rgb.shape != (480,640,3):
                        raise RuntimeError('相机帧尺寸异常，停止该条录制')
                    rgb = cv2.resize(rgb, (320,240), interpolation=cv2.INTER_AREA)
                    ok, jpeg = cv2.imencode('.jpg', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                    if ok: image['jpeg'] = jpeg.tobytes()
                    if previous is not None:
                        old_joints, old_rgb = previous
                        dataset.add_frame({'observation.state':old_joints, 'action':joints,
                                           'observation.images.external_front':old_rgb, 'task':TASK})
                        count += 1
                        state['frames_this_episode'] = count
                    previous = (joints, rgb.copy())
                    timing.append({'elapsed_s':time.monotonic()-started,'raw':raw})
                    time.sleep(max(0, .1-(time.monotonic()-tick)))
                state['mode'] = 'saving'
                if discard.is_set() or count < 20:
                    dataset.clear_episode_buffer()
                    state['message'] = '该条已丢弃。少于2秒的片段不会用于训练。'
                else:
                    index = state['episodes']
                    dataset.save_episode()
                    (root/f'capture-timing-{index:03d}.json').write_text(json.dumps(timing))
                    state['episodes'] += 1
                    state['message'] = f'已保存{state["episodes"]}条。复位物体后可录下一条，或开始训练。'
        except Exception as exc:
            state['message'] = '录制失败：'+str(exc)
            print(state['message'], flush=True)
            if dataset.has_pending_frames():
                dataset.clear_episode_buffer()
        finally:
            if camera is not None and camera.is_connected:
                camera.disconnect()
            state['mode'] = 'idle'
            lock.release()

    @app.post('/record')
    def record():
        if state['mode'] != 'idle' or not lock.acquire(blocking=False):
            return JSONResponse({'message':'正在录制、保存或训练，不重复开始'}, status_code=409)
        stop_record.clear(); discard.clear()
        state['frames_this_episode'] = 0
        state['mode'] = 'preparing'
        state['message'] = '相机准备中，等显示正在录制后再操作。'
        threading.Thread(target=capture, daemon=True).start()
        return {'message':state['message']}

    @app.post('/save')
    def save():
        if state['mode'] != 'recording':
            return JSONResponse({'message':'当前不在录制'}, status_code=409)
        stop_record.set()
        return {'message':'正在结束并保存该条'}

    @app.post('/discard')
    def discard_episode():
        if state['mode'] != 'recording':
            return JSONResponse({'message':'当前不在录制'}, status_code=409)
        discard.set(); stop_record.set()
        return {'message':'正在丢弃该条'}

    @app.post('/train')
    def train():
        if state['mode'] != 'idle' or state['episodes'] < 1 or not lock.acquire(blocking=False):
            return JSONResponse({'message':'请先保存至少一条真实示范，等保存完成再训练'}, status_code=409)
        try:
            dataset.finalize()
            output.mkdir(parents=True, exist_ok=True)
            with (output/'train.log').open('a') as log:
                process = subprocess.Popen(['bash',str(ROOT/'run'),'python','-B','-u',
                    str(ROOT/'scripts/train_act_local.py'),'--dataset',str(root),'--output',str(output)],
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            trainer['process'] = process
            state['mode'] = 'training'
            state['message'] = 'ACT训练已启动，可以离开。不会运行机械臂。'
            state['training_pid'] = process.pid
            state['output'] = str(output)
            return {'message':state['message']}
        finally:
            lock.release()

    print('ACT RECORDING READY:', ORIGIN, 'DATASET:', root, flush=True)
    try:
        uvicorn.run(app, host=HOST, port=PORT, log_level='warning')
    finally:
        stop_record.set()
        # Do not finalize concurrently with the capture thread.
        if lock.acquire(timeout=10):
            try: dataset.finalize()
            finally: lock.release()


if __name__ == '__main__':
    main()
