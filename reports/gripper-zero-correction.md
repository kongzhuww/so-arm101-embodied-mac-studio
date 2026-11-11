# 用户确认后的6号零位校正

用户明确同意执行6号零位修正，不自动运行ACT。应用前暂停网页以避免串口并发，核对从臂120、所有电机标定指纹及12V供电，保存备份。

依据：旧范围2007～3438；重装后手动闭合约970、自然最大张开2400，支持报告坐标整体增加约1038。该端点匹配不是5号机械朝向或ACT实机成功的证明。

Feetech定义确认：`present = actual - homing_offset`（sign=-1），因此homing_offset从-288改为-1326。不是改成+750。只有6号EEPROM解锁、偏移写入、恢复原锁定状态；若6号尚有力矩则仅允许关闭，绝不启动力矩。本次应用前6号力矩已经为0。

结果：同一姿态995 → 2034（预期+1038，含1刻度读数差），新连接读回仍为2034、homing_offset=-1326、torque_enable=0、lock=1，范围仍2007～3438。

1～5号配置、目标和力矩状态均逐项核对不变。没有目标位置写入或自动开合动作。原示教点、数据及模型共125个受保护文件SHA256不变。

同步更新 `calibration/so101-follower-20261031T144217-014032Z.json` 中仅gripper.homing_offset字段为-1326；所有range_min/range_max不变，模型的关节归一化/逆归一化数值定义不变。原文件备份在 `backups/follower-calibration-before-gripper-zero.json`。不要只把该文件恢复旧值而不同时核对实际硬件。

完整硬件备份：`backups/follower6-before-zero-20261108T171720048704Z.json`。
写入日志：同名 `.writes.json`。
应用报告：`reports/gripper-zero-repair-20261108T171720048704Z.json`。
独立连接读回：`reports/gripper-zero-fresh-readback.json`。
脚本：`scripts/repair_gripper_zero.py`；默认仅离线计划，不访问串口，已应用后基线检查会拒绝再次应用。
测试：本轮55项离线测试通过，包括只允许6号零位/关力矩/锁定、无目标写入、供电不足禁止写入、读回失败不重试不启动力矩。

网页已恢复，摄像头、所选模型和示教版本状态保留。下一步仅验证复位时的物理夹爪姿态，不应直接把读回成功当作整套ACT已修复。5号未作任何偏移校正，机械朝向仍需实物对照。
