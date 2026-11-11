# Staged SO-101 calibration plan — NOT executed

## Confirmed baseline
User reported connecting 12V without observed motion. One authorized raw read-only probe completed: model_number=777 for IDs 1..6, torque_enable=0 for every motor at the sampling times. Port closed. Raw positions 1800,1867,488,1077,795,1789. This is not a calibration and does not prove future torque state or power-up safety.

## Stock entrypoint hazard
Pinned src/lerobot/scripts/lerobot_calibrate.py calls device.connect(calibrate=False) before device.calibrate(). SOFollower.connect still calls configure(). configure uses bus.torque_disabled(): disables torque on entry, then always enables it on exit, including exceptions. Thus even the stock calibration command can enable torque before its manual-motion stage. DO NOT RUN the stock command for this first calibration.

## Actual calibration writes
SerialMotorsBus.record_calibration():
1. disable_torque: writes Torque_Enable=0 and, on STS, Lock=0. May let an unsupported arm drop if state changed since baseline.
2. set_operating_mode('position'): writes mode register (no position goal).
3. Prompt user to place joints at reference mid-range pose.
4. set_half_turn_homings(): first resets offsets and full-range position limits, reads positions, calculates offsets and writes Homing_Offset. Changes interpretation of encoder readings; do not reuse stale goals.
5. User manually moves selected joints through safe collision-free ranges while positions are read. The stock follower gives wrist_roll full encoder range; not an instruction to physically twist cables a full turn.
6. Writes final Homing_Offset and min/max limits, saves calibration JSON. Torque must remain off in the staged implementation.

No goal-position movement is needed for this calibration, but torque changes and manual handling can still cause physical motion. Do not enable torque simply to check whether a new calibration works. That is a separate approved phase.

## Proposed gated stages
A. Before writes: add a SEPARATE read-only backup tool with its own explicit register allowlist; do not silently broaden the audited 18-packet diagnostic. Back up model/ID, torque, lock, mode, phase, homing offset and position limits, firmware if supported. Save immutable local snapshot. Read and recheck torque before manual handling; on alarm or unexpected state, stop.
B. Explain reference pose using official SO101 calibration diagram/video. User must be present and support the arm; no forcing gears, no pinching fingers, no contact with hard stops, and no blind full turns of wrist/cables. Camera helps verify pose but cannot establish mechanical limits.
C. After separate explicit write authorization: staged helper using the bus directly, not SOFollower.connect or torque_disabled context manager. Fixed write allowlist excludes ID, baudrate, goals, positive torque-enable and PID/acceleration settings. No automatic reuse of an old calibration file. On errors, no torque-on cleanup and no automatic rollback; report any completed writes and close port. Backup permits deliberate recovery, not an unsafe automatic restore.
D. Reference capture and manual range collection require explicit user checkpoints, and suitable noninteractive session state or a foreground terminal. Do not feed fake ENTER inputs to skip human steps. Use finite collection windows if implemented for this harness.
E. Review results and save project calibration; verify schema against this pinned LeRobot. Re-read values and confirm torque remains off. Still no autonomous motion authorized.

## Current boundary
User chose the official calibration logic with a safe entrypoint that bypasses connection-time torque enabling. Implemented SEPARATE scripts/backup_motor_settings.py, validated 11 new tests plus 23 existing tests (34 total), and cross-checked all 12 allowed register addresses/sizes against installed rustypot STS3215 metadata. Ran one authorized read-only snapshot after the selection. No motor register writes.

Snapshot: backups/motor-settings-20261031T134617-802138Z.json. All six motors: model777, torque0 (also rechecked at each motor's end), lock1, mode0(position), phase12 (bit4 clear), raw homing offset85, min0/max4095, firmware3.10. This is a backup, NOT evidence that LeRobot manual calibration is complete. Port closed.

Implemented scripts/safe_official_calibrate.py: directly calls official bus.record_calibration, without SOFollower.connect/configure. Restricted transport allows only torque OFF, lock OPEN, position mode, homing offset and limits; blocks goal, torque-on, ID, baudrate and all sync writes. Baseline checked against explicit backup; real TTY and human CALIBRATE before writes / SAVE before final limits required. Journals attempted writes, reads back and saves official schema JSON, closes without torque changes. 16 new offline tests; 50 total pass, including actual official algorithm with mocked transport. See MANUAL-CALIBRATION.md.

Attempt to open a dedicated Terminal for read-only preflight and waiting at human consent via osascript timed out after25s. No retry. Subsequent check found no calibration Python process and no calibration journal directory. Launch/automation permission is unresolved; do not assume the GUI request could not appear later. No observed hardware calibration run or register write. Necessary calibration writes still require separate human confirmation; do not type CALIBRATE before reviewing the reference/manual workflow. Keep stable physical support.

Official reference: https://huggingface.co/docs/lerobot/main/en/so101#calibrate
