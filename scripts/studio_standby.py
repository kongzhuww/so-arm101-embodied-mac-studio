"""Explicit one-shot standby motion. Import/target selection never accesses hardware."""
import json
import math
import time
from pathlib import Path


def validate_target(values):
    if not isinstance(values,list) or len(values)!=6 or any(type(v)is not int or not 0<=v<=4095 for v in values):
        raise ValueError('待机点必须是6个有效舵机刻度')
    return list(values)


def model_standby(model_path):
    """Use training snapshots, NOT whichever teaching version is selected now."""
    manifest=json.loads((Path(model_path).parent/'run-manifest.json').read_text())
    sources=manifest.get('source',[])
    if isinstance(sources,str):sources=[sources]
    poses=[]
    for source in sources:
        for file in sorted((Path(source)/'metadata').glob('episode_*.json')):
            points=json.loads(file.read_text()).get('version',{}).get('waypoints',{})
            if 'p1_standby' in points:
                pose=validate_target(points['p1_standby'])
                if pose not in poses:poses.append(pose)
    if len(poses)!=1:
        raise ValueError('该模型缺少唯一的示教待机快照；未发送动作，请选择修正版模型')
    return poses[0]


def run_standby(target,bus_factory,stop_event,emit=lambda message:None,*,clock=time.monotonic,wait=None):
    target=validate_target(target);wait=wait or stop_event.wait
    result={'status':'error','target':target,'held':False};bus=None
    def read():
        positions=[]
        for motor in range(1,7):
            value,error=bus.read_register_with_error(motor,'present_position',retries=1)
            if error:raise RuntimeError(f'{motor}号读取失败：{error}')
            positions.append(int(value)&4095)
        return positions
    def write(motor,register,value):
        error=bus.write_register_with_error(motor,register,int(value),retries=0)
        if error:raise RuntimeError(f'{motor}号{register}写入失败：{error}')
    try:
        if stop_event.is_set():
            result.update(status='stopped',message='已取消复位，未连接机械臂');return result
        bus=bus_factory();start=read()
        for motor,limit in enumerate([300,450,450,300,200,130],1):
            for register,value in [('goal_position',start[motor-1]),('torque_limit',limit),('goal_speed',300),('acceleration',5),('goal_time',0)]:
                write(motor,register,value)
        for motor in range(1,7):
            if stop_event.is_set():break
            write(motor,'torque_enable',1)
        duration=max(3.0,max(abs(a-b)for a,b in zip(start,target))/180)
        steps=math.ceil(duration*20);began=clock()
        emit(f'正在平滑回待机位，约{duration:.1f}秒；不会启动ACT')
        for step in range(1,steps+1):
            if stop_event.is_set():
                result.update(status='stopped',message='复位已停止，未继续前往待机');return result
            alpha=(1-math.cos(math.pi*step/steps))/2
            for motor,(a,b) in enumerate(zip(start,target),1):
                if stop_event.is_set():
                    result.update(status='stopped',message='复位已停止');return result
                write(motor,'goal_position',round(a+(b-a)*alpha))
            wait(max(0,began+duration*step/steps-clock()))
        deadline=clock()+3
        while True:
            if stop_event.is_set():
                result.update(status='stopped',message='复位已停止');return result
            positions=read();result['positions']=positions
            if all(abs(a-b)<=20 for a,b in zip(positions,target)):
                result.update(status='completed',message='已确认到达待机位（各关节误差≤20刻度），保持力矩；未启动ACT');return result
            if clock()>=deadline:
                result.update(status='timeout',message='待机复位未在期限内到位，已停止；请检查路径/负载，勿直接启动ACT');return result
            wait(.1)
    except Exception as exc:
        result.update(status='error',message=f'待机复位失败：{exc}')
    finally:
        if bus is not None:
            try:
                current=read()
                for motor,value in enumerate(current,1):write(motor,'goal_position',value)
                result.update(held=True,positions=current)
                result['message']=result.get('message','复位结束')+'；已保持实测位置'
            except Exception as exc:
                result['message']=result.get('message','复位结束')+f'；未能确认保持：{exc}'
                if result['status']=='completed':result['status']='error'
            try:bus.close()
            except Exception:pass
    return result
