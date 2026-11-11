"""One selected motor, +/-16 raw counts once. No hardware access on import.

Uses existing fixed low RAM settings, treats a goal write as actuation, and
always attempts selected-motor torque-off afterward. No explicit torque-on,
EEPROM writes, motion retry, automatic return, or writes to other motors.
"""
import time
import gripper_microstep as m

SETTINGS = {'torque_limit': 50, 'acceleration': 1, 'goal_speed': 50, 'goal_time': 0}


class RangeStop(m.Stop):
    pass


def perform(raw, calibration, journal, motor_id, step, bounds,
            stopped=lambda: False, clock=time.monotonic, sleep=time.sleep):
    if type(motor_id) is not int or motor_id not in range(1, 7):
        raise m.Stop('请选择1至6号电机')
    if type(step) is not int or step not in (-16, 16):
        raise m.Stop('只允许正负16格单步')
    if stopped():
        raise m.Stop('控制已停用')
    gate = m.Motor6Gate(raw, journal)  # READ allowlist only.
    baseline = m.check_baseline(gate, calibration)
    start = baseline[motor_id]
    target = start+step
    c = next(c for c in calibration.values() if c.id == motor_id)
    lo, hi = bounds[motor_id]
    if not max(lo, c.range_min+64) <= min(start, target) <= max(start, target) <= min(hi, c.range_max-64):
        raise RangeStop('到达本轮小范围边界，请反向；不会自动扩大范围')
    for name, value in [('firmware_major_version', 3), ('firmware_minor_version', 10),
                        ('velocity_unit_factor', 50), ('acceleration_multiplier', 1),
                        ('maximum_acceleration', 50)]:
        if gate.read(motor_id, name) != value:
            raise m.Stop('电机固件或缩放设置与已核对参数不一致')
    if gate.read(motor_id, 'status') or not 100 <= gate.read(motor_id, 'present_voltage') <= 130 or gate.read(motor_id, 'present_temperature') > 45:
        raise m.Stop('电机状态、电压或温度异常')

    def stable_off():
        if stopped():
            raise m.Stop('控制已停用')
        for i in range(1, 7):
            if gate.read(i, 'torque_enable') != 0 or abs(gate.read(i, 'present_position')-baseline[i]) > 2:
                raise m.Stop('操作前电机不静止或扭矩未关闭')

    def write(name, value):
        if not ((name in SETTINGS and value == SETTINGS[name]) or
                (name == 'goal_position' and value == target) or
                (name == 'torque_enable' and value == 0)):
            raise m.Stop('禁止的电机写入')
        journal.emit(event='write_attempt', id=motor_id, register=name, value=value)
        status = raw.write_register_with_error(motor_id, name, value, retries=0)
        journal.emit(event='write_response', id=motor_id, register=name, status=status)
        if status:
            raise m.Stop('写入返回报警，不重试动作')

    stable_off()
    sleep(.1)
    stable_off()
    reached = False
    try:
        for name, value in SETTINGS.items():
            write(name, value)
            stable_off()
        for name, value in SETTINGS.items():
            if gate.read(motor_id, name) != value:
                raise m.Stop('低出力设置读回不匹配')
        stable_off()
        journal.emit(event='first_goal_is_actuation', id=motor_id, start=start, target=target)
        write('goal_position', target)
        settled = 0
        deadline = clock()+2
        while clock() < deadline:
            if stopped():
                raise m.Stop('已请求停用，关闭当前电机扭矩')
            pos = gate.read(motor_id, 'present_position')
            torque = gate.read(motor_id, 'torque_enable')
            alarm = gate.read(motor_id, 'status')
            temp = gate.read(motor_id, 'present_temperature')
            limit = gate.read(motor_id, 'torque_limit')
            journal.emit(event='sample', id=motor_id, position=pos, torque=torque,
                         status=alarm, temperature=temp, torque_limit=limit)
            if alarm or abs(pos-start) > 20 or temp > 45 or not 0 <= limit <= 50 or torque not in (0, 1):
                raise m.Stop('电机状态或位移异常，停止')
            for i in range(1, 7):
                if i == motor_id:
                    continue
                if gate.read(i, 'torque_enable') != 0 or abs(gate.read(i, 'present_position')-baseline[i]) > 16:
                    raise m.Stop('其他电机状态变化，停止')
            settled = settled+1 if abs(pos-target) <= 2 else 0
            if settled >= 3:
                reached = True
                break
            sleep(.05)
        position = gate.read(motor_id, 'present_position')
        journal.emit(event='result', reached=reached, id=motor_id, start=start,
                     target=target, position=position)
    finally:
        off = False
        for _ in range(2):
            try:
                write('torque_enable', 0)
                if gate.read(motor_id, 'torque_enable') == 0:
                    off = True
                    break
            except Exception as exc:
                journal.emit(event='disable_problem', error=type(exc).__name__)
        journal.emit(event='cleanup', id=motor_id, torque_off_verified=off)
        if not off:
            raise m.Stop('无法确认扭矩关闭，请立即手动断电')
    if not reached:
        raise m.Stop('本次未到位，已关闭扭矩；不自动重试或加大出力')
    return {'motor': motor_id, 'start': start, 'target': target, 'position': position,
            'reached': reached, 'torque_off': True}
