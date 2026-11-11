# First powered test: NOT executed or authorized by offline validation

## Offline results
- config/offline-validation.json stores the explicit calibration_dir/id, camera parameters and file fingerprints. This is an offline manifest, not a config to pass to stock robot connection scripts.
- scripts/validate_local_config.py decodes official RobotConfig and MotorCalibration data without creating a Robot or Camera. Its OfflineBus can only perform numeric conversions; connect/write methods refuse.
- 77 total tests pass, including forbidding serial, rustypot.Bus, Robot and camera construction during report generation.
- reports/offline-validation-20261031-01.json records all six joint conversions and planning blockers. Calibration file fingerprint and final hardware snapshot match.

## Findings that block immediate torque enable
Historical snapshot (not live): elbow3103 near/just past recorded max3102; gripper2019 near min2007. These are not safe evidence for initial powered motion. A proposed64-count inward margin would exclude both; it is a planning heuristic, not a mechanically certified margin and has NOT been written anywhere on the arm.

Wrist roll's recorded0..4095 is an encoder range, not permission to rotate through cables. Exclude it from the first powered test until its cable-safe interval is checked.

Official DEGREES normalization uses each measured min/max midpoint as0; it does NOT necessarily mean the2047 captured reference posture or URDF zero. Gripper normalized values are0..100 percent, NOT degrees. Body degree conversion does not clamp goals to limits. Thus max_relative_target=1.0 alone is NOT sufficient protection, and IK compatibility must be validated separately.

## Next small step
1. Obtain fresh read-only torque/position information and inspect scene; user confirms support, clear workspace and cutoff. Do not infer current state from the historical report.
2. Only after confirming torque off, user gently supports and moves joints away from endpoints. Do not force high-ratio gears, twist cables, or blindly target numeric midpoints. Read-only recheck afterward. No automatic positioning.

## Before any later powered motion (separate explicit approval)
- Select ONE visible, collision-free joint only after inspecting supported posture. Keep gripper and wrist-roll excluded initially unless specifically reviewed.
- Audit/prepare goal preload to fresh measured position while torque is off, so enabling does not chase an old stored goal. Goal writes themselves require authorization, even if torque is off.
- Review actual firmware/register units and conservative speed/acceleration/torque settings; do not guess their units or reuse stock defaults.
- Enforce raw soft intervals, bounded small displacement (initially no more than about1degree, subject to scene), bounded duration and feedback. No stock Robot.connect/configure, homing, phone IK or unfamiliar model.
- Define stop/cleanup with stable physical support. Turning torque off can drop a loaded arm; do not improvise this only after an error. User remains at the cutoff.
- Powered test success is not validation of grasping, calibration pose semantics, phone teleoperation or autonomous policy execution.
