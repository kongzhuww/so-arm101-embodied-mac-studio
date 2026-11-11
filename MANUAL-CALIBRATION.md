# 本次手动校准（尚未实机执行写入）

入口：scripts/safe_official_calibrate.py。采用已安装的官方 SerialMotorsBus.record_calibration()；不创建 Robot 对象、不调用普通 configure()。

## 保护与边界
- 默认只显示计划，不接串口。硬件运行要求真实前台终端、两个确认参数、指定备份文件。
- 先只读比对备份与当前模式/偏移/范围/扭矩等；不一致就停，不自动恢复。
- 在真人键入 CALIBRATE 前不允许写寄存器。
- 唯一允许的写入：Torque_Enable=0、Lock=0、Operating_Mode=0、Homing_Offset(-2047..2047)、Min/Max_Position_Limit(0..4095)。
- 禁止使能扭矩、目标位置、改ID/波特率、PID、加速度和所有同步写。每次非扭矩/锁写入前重新检查该舵机扭矩为0；读写响应有任何报警就停，无自动重试。
- 官方算法先写参考偏移，之后才采范围。因此中途取消也可能已有部分改动；不会自动回滚或重新使能扭矩，必须查看日志。不是电气/机械安全保证。

## 前台终端阶段（需要人参与）
1. 保持支撑、清空活动区、手离关节缝隙、电源可随手切断。只读预检后先停在 CALIBRATE 提示。未明确同意写校准参数前不要输入。
2. 确认后输入 CALIBRATE。此时会写扭矩关闭、解锁和位置模式，不发运动目标。若出现自行运动/异常声音，立即断电并停止。
3. 接下来官方提示将关节摆到参考中位。参考官方图示/视频： https://huggingface.co/docs/lerobot/main/en/so101#calibrate 。不要凭记忆猜各轴方向；请与助手核对后再在这个提示处按回车。不要强扳高减速比舵机；难以移动或结构干涉时停下。
4. 回车会重设并写入参考偏移及暂时的范围，然后开始读取位置。逐个缓慢手动覆盖需要记录的无碰撞可用活动范围，不能硬顶限位。wrist_roll 按官方算法直接记录全编码器范围，不需要物理旋转整圈，更不能拧绕线缆。手动摆位本身会让机械臂运动，支撑其他臂段以防掉落。
5. 五个需采范围的关节（shoulder_pan、shoulder_lift、elbow_flex、wrist_flex、gripper）均完成后按一次回车停止记录。程序检查各关节范围非零且包含参考点，显示候选JSON。
6. 先检查候选范围，再输入 SAVE 才写最终偏移/限位并保存JSON。任何其他输入取消最终保存，但之前的参考写入仍然存在。
7. 成功后会读回校准与全六轴扭矩状态、保存项目 calibration/ 下的JSON、关闭串口。不会使能扭矩。文件是官方 MotorCalibration 字段格式；后续使用需要显式指定同一 calibration_dir/id，不会自动加载本次时间戳文件。

## 可安全执行的离线命令
```bash
cd /Users/kongzhu/so-arm101-act
bash run python -B scripts/safe_official_calibrate.py
bash run python -B -m unittest discover -s scripts -p 'test_*.py' -v
```
50项离线测试通过，包括在假串口上调用真实官方校准算法。测试不是实机校准已完成的证明。实际写入仍待真人确认。
