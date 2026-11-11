"""Local phone joint selector. Actual motion only with --execute --ack-joint-motion.

A page GET, selection change or status refresh never moves a motor. Each POST
moves one selected motor by exactly +/-16 counts; no queue or auto-repeat.
Serves only on the phone hotspot interface; phone IP/origin restricted.
"""
import argparse
from contextlib import closing
from datetime import datetime, timezone
import threading
import time
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

import joint_step as joint

HOST = '10.19.220.84'
PORT = 4455
ORIGIN = f'http://{HOST}:{PORT}'
CLIENTS = {'10.19.220.45', HOST, '127.0.0.1'}
HTML = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>电机单步控制</title><style>
body{background:#111827;color:#f8fafc;font:18px system-ui;max-width:560px;margin:24px auto;padding:16px}
select,button{font:inherit;padding:18px;border-radius:12px;border:0;margin:8px 0;width:100%}
.row{display:flex;gap:16px}.row button{font-size:34px;background:#2563eb;color:white;height:100px}
button:disabled{opacity:.35}#stop{background:#b91c1c;color:white}pre{white-space:pre-wrap;font-size:16px;background:#1e293b;padding:16px;border-radius:12px}
p{color:#cbd5e1;line-height:1.5}</style>
<h2>选电机 → 点一次走一步</h2>
<p>会真正运动。每次约1.4°，随后关闭扭矩。手离开关节，臂段保持支撑，随时可断电。</p>
<select id="motor"><option value="1">1号 · 底座旋转</option><option value="2">2号 · 肩部</option><option value="3">3号 · 肘部</option><option value="4">4号 · 手腕俯仰</option><option value="5">5号 · 手腕旋转</option><option value="6">6号 · 夹爪</option></select>
<div class="row"><button id="minus">− 一步</button><button id="plus">＋ 一步</button></div>
<button id="stop">停用控制</button><pre id="result">正在连接…</pre>
<p>切换电机不会运动。每次只点一下，不要连点；＋／−表示编码方向，不保证是你预期的左右或开合。长按不重复。</p>
<p>本轮：1–4、6号各限起点±64格（约5.6°）；5号限±16格。低出力不够时会停，不自动加力。页面按钮不是安全级急停，异常立即物理断电。</p>
<script>
const result=document.querySelector('#result');let busy=false,uncertain=false;
const controls=['#minus','#plus','#motor'].map(x=>document.querySelector(x));
function lock(value){controls.forEach(x=>x.disabled=value)}
async function status(){if(uncertain){lock(true);return;}try{const r=await fetch('/status',{cache:'no-store'});const s=await r.json();
 if(!busy){lock(s.disabled);result.textContent=s.message+'\\n位置：'+JSON.stringify(s.positions)+'\\n剩余：'+s.remaining_seconds+'秒';}}
 catch(e){if(!busy){lock(true);result.textContent='连接断开，不要重复提交动作；检查Pi。'}}}
async function step(delta){if(busy)return;busy=true;lock(true);result.textContent='正在执行一次小步…';
 try{const r=await fetch('/step',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({motor:Number(document.querySelector('#motor').value),delta,request_id:Date.now()+'-'+Math.random()})});
 const s=await r.json();result.textContent=s.message;if(!r.ok)alert(s.message);}
 catch(e){uncertain=true;result.textContent='动作结果不明，请勿重试；查看Pi或物理断电。';alert(result.textContent);}
 finally{busy=false;await status();}}
document.querySelector('#minus').onclick=()=>step(-16);
document.querySelector('#plus').onclick=()=>step(16);
document.querySelector('#stop').onclick=async()=>{lock(true);try{await fetch('/stop',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'});await status();}catch(e){alert('停止请求未送达，请物理断电。')}};
setInterval(status,1000);status();
</script></html>'''


def build_app(calibration, positions, opener=joint.m.open_raw, lease=600):
    app = FastAPI()
    operation = threading.Lock()
    stopped = threading.Event()
    deadline = time.monotonic()+lease
    bounds = {i: (p-(16 if i == 5 else 64), p+(16 if i == 5 else 64)) for i, p in positions.items()}
    state = {'positions': dict(positions), 'message': '已就绪：选择电机后，点＋或−走一步', 'disabled': False}
    seen = set()

    @app.middleware('http')
    async def restrict(request: Request, call_next):
        if request.client.host not in CLIENTS:
            return JSONResponse({'message': '只允许手机热点客户端'}, status_code=403)
        if request.method == 'POST' and (request.headers.get('origin') != ORIGIN or
                request.headers.get('content-type', '').split(';')[0] != 'application/json'):
            return JSONResponse({'message': '禁止跨站或非JSON控制请求'}, status_code=403)
        return await call_next(request)

    @app.get('/', response_class=HTMLResponse)
    def index():
        return HTMLResponse(HTML, headers={'Cache-Control': 'no-store'})

    @app.get('/status')
    def status():
        if time.monotonic() >= deadline:
            stopped.set()
        return {**state, 'disabled': stopped.is_set(),
                'remaining_seconds': max(0, int(deadline-time.monotonic()))}

    @app.post('/stop')
    def stop():
        stopped.set()
        state['message'] = '已停用控制；若动作正在执行，将尝试关闭扭矩。异常请物理断电。'
        return {'message': state['message']}

    @app.post('/step')
    async def step(request: Request):
        data = await request.json()
        if not isinstance(data, dict):
            return JSONResponse({'message': '无效请求'}, status_code=400)
        motor, delta, request_id = data.get('motor'), data.get('delta'), data.get('request_id')
        if type(motor) is not int or motor not in range(1, 7) or type(delta) is not int or delta not in (-16, 16) or not isinstance(request_id, str) or not 5 <= len(request_id) <= 80:
            return JSONResponse({'message': '无效电机、步长或请求编号'}, status_code=400)
        if stopped.is_set() or time.monotonic() >= deadline:
            return JSONResponse({'message': '本轮控制已停用或超时'}, status_code=423)
        if not operation.acquire(blocking=False):
            return JSONResponse({'message': '正在执行上一个动作，不排队'}, status_code=409)
        if request_id in seen:
            operation.release()
            return JSONResponse({'message': '重复动作已拒绝，不重试'}, status_code=409)
        seen.add(request_id)
        if len(seen) > 60:
            stopped.set()
            operation.release()
            return JSONResponse({'message': '本轮60次操作已用完'}, status_code=423)
        # Run blocking serial work off the event loop so /stop stays available.
        from starlette.concurrency import run_in_threadpool
        return await run_in_threadpool(execute, motor, delta)

    def execute(motor, delta):
        journal = None
        try:
            path = joint.m.ROOT/'motion-tests'/('manual-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S-%fZ')+'-'+uuid.uuid4().hex[:6]+'.jsonl')
            journal = joint.m.safety.Journal(path)
            with closing(opener()) as raw:
                result = joint.perform(raw, calibration, journal, motor, delta, bounds,
                                       stopped=lambda: stopped.is_set() or time.monotonic() >= deadline)
                # Read all six independently after selected-motor torque-off.
                gate = joint.m.Motor6Gate(raw, journal)
                for i in range(1, 7):
                    if gate.read(i, 'torque_enable') != 0:
                        raise joint.m.Stop('有电机扭矩未关闭，请立即断电')
                    state['positions'][i] = gate.read(i, 'present_position')
            journal.emit(event='port_closed', result=result)
            state['message'] = f"{motor}号完成：{result['start']} → {result['position']}，六轴扭矩已关闭"
            return {'message': state['message'], **result}
        except joint.RangeStop as exc:
            state['message'] = str(exc)
            return JSONResponse({'message': str(exc)}, status_code=400)
        except Exception as exc:
            stopped.set()
            state['message'] = '已停用：'+str(exc)+'。不要重复动作，检查Pi；异常请物理断电。'
            print(state['message'], flush=True)
            return JSONResponse({'message': state['message']}, status_code=423)
        finally:
            if journal is not None:
                journal.close()
            operation.release()

    app.state.stop_event = stopped
    return app


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--ack-joint-motion', action='store_true')
    a = p.parse_args(argv)
    if a.execute != a.ack_joint_motion:
        p.error('Both flags require current selected-joint motion consent')
    if not a.execute:
        print('PLAN ONLY: no port, listener, settings or motion. Phone selector1..6; +/-16 once; torque off afterward.')
        return 0
    calibration = joint.m.load_calibration()
    with closing(joint.m.open_raw()) as raw:
        class Log:
            def emit(self, **kw): pass
        positions = joint.m.check_baseline(joint.m.Motor6Gate(raw, Log()), calibration)
    app = build_app(calibration, positions)
    server = uvicorn.Server(uvicorn.Config(app, host=HOST, port=PORT, log_level='warning'))
    def expire():
        app.state.stop_event.set()
        server.should_exit = True
    timer = threading.Timer(600, expire)
    timer.daemon = True
    timer.start()
    print(f'JOINT CONTROL READY: {ORIGIN}/ | 10 minutes | no motion until button POST', flush=True)
    try:
        server.run()
    finally:
        app.state.stop_event.set()
        timer.cancel()


if __name__ == '__main__':
    raise SystemExit(main())
