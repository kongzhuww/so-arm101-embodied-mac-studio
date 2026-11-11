"""Phone-only WebXR probe. No robot/serial/camera imports or connections.

Run with the project runner; serves the installed teleop frontend on the hotspot
interface for up to 20 minutes. Browser capability reports and received poses
are diagnostic data only, never motor commands.
"""
import json
import logging
import math
from pathlib import Path
import threading
import time

from fastapi import Request
from fastapi.responses import HTMLResponse
import teleop
import uvicorn

HOST = "10.19.220.84"
PORT = 4443
STATE_PATH = Path(__file__).resolve().parents[1] / "reports/phone-input-status.json"
STATE_PATH.parent.mkdir(exist_ok=True)
state = {"robot_connected": False, "started_at": time.time(), "pose_count": 0}
lock = threading.Lock()


def save_state():
    tmp = STATE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    tmp.replace(STATE_PATH)


def receive_pose(pose, message):
    with lock:
        state["pose_count"] += 1
        state["last_pose_at"] = time.time()
        state["move_pressed"] = bool(message.get("move"))
        state["button_a"] = message.get("reservedButtonA") is True
        state["button_b"] = message.get("reservedButtonB") is True
        state["phone_pose_flu"] = pose.tolist()
        # Raw WebXR position is independent of the Move button.
        position = message.get("position", {})
        state["raw_position_m"] = {
            k: v for k in ("x", "y", "z")
            if isinstance((v := position.get(k)), (int, float)) and math.isfinite(v)
        }
        if state["pose_count"] == 1 or state["pose_count"] % 30 == 0:
            save_state()
            print("PHONE_INPUT " + json.dumps(state), flush=True)


probe = teleop.Teleop(host=HOST, port=PORT)
probe.subscribe(receive_pose)
# Reuse the installed package's routes and WebXR protocol, with a diagnostic
# extension only. This app has no arm objects and no action-sending code.
app = probe._Teleop__app
package_dir = Path(teleop.__file__).parent
capability_script = """
<script>
const reportEvent = (event, detail) => fetch('/diagnostics', {
  method:'POST', headers:{'Content-Type':'application/json'},
  body:JSON.stringify({event, detail:String(detail).slice(0,2000)})
}).catch(() => {});
window.addEventListener('error', e => reportEvent('javascript-error', e.message));
window.addEventListener('unhandledrejection', e => reportEvent('promise-error', e.reason));
const start = document.querySelector('#start-button');
if (start) start.addEventListener('click', () => reportEvent('start-click', 'clicked'), true);
const log = document.querySelector('#log');
if (log) new MutationObserver(() => reportEvent('page-message', log.textContent))
  .observe(log, {childList:true, characterData:true, subtree:true});
if (navigator.xr) {
  const originalRequest = navigator.xr.requestSession.bind(navigator.xr);
  navigator.xr.requestSession = function(...args) {
    reportEvent('xr-request', JSON.stringify(args));
    const result = originalRequest(...args);
    result.then(() => reportEvent('xr-started', 'session created'),
      e => reportEvent('xr-start-failed', e.name + ': ' + e.message));
    return result;
  };
}
(async () => {
  const report = {
    secureContext: window.isSecureContext,
    userAgent: navigator.userAgent,
    webxr: !!navigator.xr,
    immersiveAR: false
  };
  try {
    report.immersiveAR = !!navigator.xr &&
      await navigator.xr.isSessionSupported('immersive-ar');
  } catch (e) { report.error = String(e); }
  await fetch('/diagnostics', {method:'POST',
    headers:{'Content-Type':'application/json'}, body:JSON.stringify(report)});
  const banner = document.createElement('div');
  banner.textContent = 'PHONE INPUT — CHECK PI BEFORE MOTOR TEST | WebXR: ' +
    report.webxr + ' | AR: ' + report.immersiveAR;
  banner.style.cssText = 'position:fixed;top:0;left:0;right:0;background:#163b24;color:white;padding:10px;font:13px sans-serif;z-index:99999;pointer-events:none';
  document.body.appendChild(banner);
})();
</script>
"""
# Handheld AR does not need floor detection or VR/hand-tracking features.
# Use the standard local reference space and only the touchscreen overlay.
html = (package_dir / "index.html").read_text().replace(
    "'local-floor'", "'local'"
).replace(
    "optionalFeatures: ['hand-tracking', 'unbounded', 'dom-overlay', 'dom-overlay-for-handheld-ar', 'hit-test', 'high-fixed-foveation-level']",
    "optionalFeatures: ['dom-overlay']"
).replace("</body>", capability_script + "</body>")
# Replace only the frontend route, preserving the installed WebSocket route.
app.router.routes[:] = [r for r in app.router.routes if getattr(r, "path", None) != "/"]


@app.get("/", response_class=HTMLResponse)
async def index():
    return html


@app.post("/diagnostics")
async def diagnostics(request: Request):
    report = await request.json()
    with lock:
        if "event" in report:
            events = state.setdefault("events", [])
            events.append({"at": time.time(), "event": str(report["event"])[:80],
                           "detail": str(report.get("detail", ""))[:2000]})
            state["events"] = events[-30:]
        else:
            state["browser"] = {k: report[k] for k in (
                "secureContext", "userAgent", "webxr", "immersiveAR", "error"
            ) if k in report}
            state["browser_report_at"] = time.time()
        save_state()
    print("BROWSER_DIAGNOSTIC " + json.dumps(report), flush=True)
    return {"ok": True, "robot_connected": False}


@app.get("/input-status")
async def input_status():
    with lock:
        return dict(state)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    with lock:
        save_state()
    server = uvicorn.Server(uvicorn.Config(
        app, host=HOST, port=PORT,
        ssl_certfile=str(package_dir / "cert.pem"),
        ssl_keyfile=str(package_dir / "key.pem"),
        log_level="info",
    ))
    timer = threading.Timer(1200, lambda: setattr(server, "should_exit", True))
    timer.daemon = True
    timer.start()
    print(f"PHONE INPUT ONLY: https://{HOST}:{PORT} (20 minute limit)", flush=True)
    try:
        server.run()
    finally:
        timer.cancel()
