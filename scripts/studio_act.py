"""ACT model catalog and one web-authorized run. No hardware is opened at import.

The web runner uses the Studio's existing camera feed, never a second camera.
Selection/loading does not enable motors. Only run_once can access the bus.
"""
from pathlib import Path
import json
import time

from teaching_versions import atomic_json

REQUIRED = ('config.json', 'model.safetensors', 'policy_preprocessor.json', 'policy_postprocessor.json')
TORQUE_LIMITS = [300, 450, 450, 300, 200, 130]


class ActModels:
    def __init__(self, root):
        self.root = Path(root)
        self.settings = self.root / 'config/act_studio.json'
        self.selection = json.loads(self.settings.read_text()) if self.settings.exists() else {}

    def list(self):
        models = []
        output = (self.root / 'outputs').resolve()
        for final in sorted(output.glob('act_*/final'), reverse=True):
            if not final.resolve().is_relative_to(output) or not all((final / f).is_file() for f in REQUIRED):
                continue
            try:
                config = json.loads((final / 'config.json').read_text())
                if config.get('type', 'act') != 'act':
                    continue
                for filename in ('policy_preprocessor.json','policy_postprocessor.json'):
                    processor=json.loads((final/filename).read_text())
                    if any(not (final / step['state_file']).is_file()
                           for step in processor.get('steps',[]) if step.get('state_file')):
                        raise ValueError('Missing processor state')
            except (OSError, ValueError, KeyError):
                continue
            episodes = frames = None
            manifest_path = final.parent / 'run-manifest.json'
            if manifest_path.exists():
                try:
                    manifest = json.loads(manifest_path.read_text())
                    dataset = Path(manifest['dataset']).resolve()
                    if dataset.is_relative_to((self.root / 'datasets').resolve()):
                        info = json.loads((dataset / 'meta/info.json').read_text())
                        episodes, frames = info['total_episodes'], info['total_frames']
                except (OSError, ValueError, KeyError):
                    pass
            ident = final.parent.name
            label = ident.removeprefix('act_') + (f' · {episodes}条' if episodes is not None else '')
            models.append({'id':ident,'label':label,'path':str(final),
                           'episodes':episodes,'frames':frames})
        return models

    def resolve(self, ident):
        for model in self.list():
            if model['id'] == ident:
                return Path(model['path'])
        raise ValueError('模型不存在或文件不完整，请重新选择')

    def current(self):
        models = self.list()
        ident = self.selection.get('model')
        if not any(m['id'] == ident for m in models):
            ident = models[0]['id'] if models else None
        mode=self.selection.get('mode','teach')
        return {'mode':mode if mode in ('teach','act') else 'teach', 'model':ident, 'models':models}

    def save(self, *, mode=None, model=None):
        current = self.current()
        chosen_mode = mode if mode is not None else current['mode']
        if chosen_mode not in ('teach','act'):
            raise ValueError('未知工作模式')
        chosen_model = model if model is not None else current['model']
        if chosen_model is not None:
            self.resolve(chosen_model)
        self.selection = {'mode':chosen_mode,'model':chosen_model}
        atomic_json(self.settings,self.selection)
        return self.current()


class PolicyBackend:
    def __init__(self, model_path, root):
        import torch
        from lerobot.policies.act import ACTPolicy
        from lerobot.policies import make_pre_post_processors
        from run_act_inference import recorded_box_poses
        self.root=Path(root)
        self.device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        self.policy=ACTPolicy.from_pretrained(str(model_path)).to(self.device).eval()
        self.policy.reset()
        visual=Path(model_path)/'visual-preprocessing.json'
        self.view=None
        if visual.exists():
            from studio_visual import BackgroundView
            self.view=BackgroundView.load(visual)
        self.pre,self.post=make_pre_post_processors(self.policy.config,pretrained_path=model_path,
            preprocessor_overrides={'device_processor':{'device':str(self.device)}})
        self.calibration=json.loads((self.root/'calibration/so101-follower-20261031T144217-014032Z.json').read_text())
        self.boxes=recorded_box_poses(Path(model_path))
        self.standbys=[]
        manifest=Path(model_path).parent/'run-manifest.json'
        sources=json.loads(manifest.read_text()).get('source',[]) if manifest.exists() else []
        if isinstance(sources,str):sources=[sources]
        for source in sources:
            snapshots=sorted((Path(source)/'metadata').glob('episode_*.json'))
            if snapshots:
                points=json.loads(snapshots[0].read_text())['version']['waypoints']
                pose=points.get('p1_standby')
                if pose and pose not in self.standbys:self.standbys.append(pose)
        if not self.standbys:
            # Completion observation only, not commanded motion or policy input.
            self.standbys=[json.loads((self.root/'config/auto_waypoints.json').read_text())['p1_standby']]

    def predict(self, positions, rgb):
        from lerobot.common.control_utils import predict_action
        from run_act_inference import normalize_joints, denormalize_joints
        if self.view is not None:rgb=self.view.transform_rgb(rgb)
        action=predict_action({'observation.state':normalize_joints(positions,self.calibration),
            'observation.images.external_front':rgb},self.policy,self.device,self.pre,self.post,
            use_amp=False,task='Pick up the red cube and place it into the box.',robot_type='so101_follower')
        return denormalize_joints(action.detach().cpu().numpy().reshape(-1),self.calibration)


def run_once(model_path, root, read_rgb, stop_event, emit=lambda **kw:None,
             *, backend_factory=None, bus_factory=None, max_seconds=60,
             clock=time.monotonic, sleep=time.sleep):
    """Only caller explicitly starting a run may invoke this function.

    Stop/completion/time-limit/error attempts to hold measured position, not
    drop torque. Stop is checked after loading and after inference, before goals.
    Returned status is NOT a claim that the object was successfully grasped.
    Factories/clock allow fully offline tests.
    """
    import numpy as np
    if not 0 < max_seconds <= 60:
        raise ValueError('单次时间上限需要在 0～60 秒内')
    bus=None
    result={'model':str(model_path),'status':'error','steps':0,'held':False}
    try:
        emit(phase='loading',message='正在加载所选 ACT；尚未驱动机械臂')
        backend=(backend_factory or PolicyBackend)(Path(model_path),Path(root))
        if stop_event.is_set():
            result.update(status='stopped',message='加载已取消，未连接机械臂')
            return result
        rgb=read_rgb()  # Validate the shared live camera before serial access.
        if bus_factory is None:
            import rustypot
            _,definition=rustypot.find_model('sts3215')
            bus_factory=lambda:rustypot.Bus('/dev/cu.usbserial-120',1000000,0.05,{i:definition for i in range(1,7)})
        bus=bus_factory()

        def read_positions():
            positions=[]
            for motor in range(1,7):
                value,error=bus.read_register_with_error(motor,'present_position',retries=1)
                if error:raise RuntimeError(f'从臂 {motor} 号读取失败：{error}')
                positions.append(int(value)&0x0FFF)
            return positions

        def write(motor,register,value):
            error=bus.write_register_with_error(motor,register,int(value),retries=0)
            if error:raise RuntimeError(f'从臂 {motor} 号 {register} 写入失败：{error}')

        def validate_goal(values):
            a=np.asarray(values,dtype=float)
            if a.shape!=(6,) or not np.isfinite(a).all() or np.any(a!=np.round(a)) or np.any(a<0) or np.any(a>4095):
                raise ValueError('模型输出不是有效的 6 个舵机目标')
            return a.astype(int).tolist()

        start=read_positions()
        first_goal=validate_goal(backend.predict(start,rgb))
        if stop_event.is_set():
            result.update(status='stopped',message='已停止；未启用电机')
            return result
        for motor,limit in enumerate(TORQUE_LIMITS,1):
            for register,value in [('goal_position',start[motor-1]),('torque_limit',limit),
                    ('goal_speed',300),('acceleration',5),('goal_time',0)]:
                write(motor,register,value)
        # All targets/settings have been preloaded before any torque enable.
        for motor in range(1,7):
            if stop_event.is_set():break
            write(motor,'torque_enable',1)
        deadline=clock()+max_seconds
        seen_box=returning=False
        returned_since=None
        boxes=getattr(backend,'boxes',[])
        standbys=getattr(backend,'standbys',[])
        emit(phase='running',message='ACT 运行中：最多60秒，停止时保持当前位置')
        for step in range(600):
            tick=clock()
            if stop_event.is_set():
                result.update(status='stopped',message='已请求停止，本轮不再预测')
                break
            if tick>=deadline:
                result.update(status='timeout',message='达到时间上限，未确认回程完成')
                break
            positions=start if step==0 else read_positions()
            at_box=any(np.all(np.abs(np.asarray(positions[:5])-pose)<=np.array([80,110,110,100,80])) for pose in boxes) if boxes else (positions[1]>2500 and positions[2]<1650)
            seen_box=seen_box or at_box
            returning=returning or (seen_box and positions[1]<1900 and positions[2]>2100)
            returned=seen_box and any(np.all(np.abs(np.asarray(positions)-pose)<=np.array([40,80,80,60,40,100])) for pose in standbys)
            returned_since=(tick if returned_since is None else returned_since) if returned else None
            if returned_since is not None and tick-returned_since>=1:
                result.update(status='completed',message='已到待机附近并稳定；抓取结果请现场确认')
                break
            goal=first_goal if step==0 else validate_goal(backend.predict(positions,read_rgb()))
            if stop_event.is_set():
                result.update(status='stopped',message='已请求停止，未发送本步预测')
                break
            if clock()>=deadline:
                result.update(status='timeout',message='达到时间上限，未发送本步预测')
                break
            if returning and goal[1]>2000 and goal[2]<2200:
                result.update(status='blocked',message='回程又转向抓取，已中止本轮，未完成正常回程')
                break
            for motor,value in enumerate(goal,1):write(motor,'goal_position',value)
            result['steps']=step+1
            if step%10==0:emit(phase='running',step=step+1,positions=positions,goal=goal,message=f'ACT 第{step+1}步')
            sleep(max(0,.1-(clock()-tick)))
        else:
            result.update(status='timeout',message='达到单次步数上限')
    except Exception as exc:
        result.update(status='error',message=f'ACT 测试失败：{exc}')
    finally:
        if bus is not None:
            try:
                current=read_positions()
                for motor,value in enumerate(current,1):write(motor,'goal_position',value)
                result['held']=True
                result['message']=result.get('message','已结束')+'；保持当前位置与力矩'
            except Exception as exc:
                result['held']=False
                result['message']=result.get('message','已结束')+f'；未能确认保持：{exc}'
            try:bus.close()
            except Exception:pass
        emit(phase=result['status'],step=result['steps'],message=result.get('message','已结束'),held=result['held'])
    return result
