# Motor6 microstep — HARDWARE EXECUTION DISABLED

**Real-hardware invalidated assumption:** Two runs logged torque_enable=1/status0 after writing Goal_Position2377 while earlier checks were torque0. No explicit torque_enable1 write was sent. Therefore the preloading phase can itself energize this device/setup; treating it as torque-off-only was incorrect. Exact firmware/library/on-wire mechanism is not yet fully established. Cleanup and independent read-only checks confirm all six torque states0 afterward. The CLI now refuses hardware execution before opening serial, even with flags/TTY.99 offline tests pass, including simulation of implicit enable, but this is NOT a successful physical motion test. Do not remove this block or simply accept torque1; redesign/review with first goal write treated as possible actuation. The following is the historical design, not currently approved for hardware use.

## Why not directly enable
Read-only diagnostic backups/gripper-settings-20261031T152325-067766Z.json found motor6 present_position2377 but goal_position0, goal_speed0, torque_limit1000, acceleration0. Blind torque-enable could pursue the stale goal. All six torque states were0. Factory scaling readbacks: velocity_unit_factor50, acceleration_multiplier1, maximum_acceleration50; firmware3.10.

## Limited test entry
scripts/gripper_microstep.py defaults to plan-only, no serial access. Hardware mode requires foreground TTY and both explicit flags, then a human MOVE6 token. Actual output is not authorized by merely opening the terminal. Port opens only for initial reads; before consent no writes are possible through the gate.

Before a human confirms: support the arm, clear the empty gripper, keep hands away from jaw and joint gaps, and have an accessible power cutoff. Moving a servo even slightly is not risk-free.

After consent:
- Recheck six model/torque/mode/calibration states, inward64-count margins except unpowered wrist roll, firmware/scaling, supply/temperature/alarm.
- Recheck gripper stability. Lock a +8count target inside its soft range; about0.7degree of the servo axis, NOT measured jaw travel.
- Write ONLY motor6 volatile RAM fields: goal_time0, goal_speed50, acceleration1, torque_limit50. These are fixed conservative RAW values. Actual speed/force have not been measured; do not treat them as certified physical limits. Torque50 is a low nominal setting, not safe human-contact force control.
- Preload goal_position to a fresh measured position with torque still off; verify all settings and preload/position. Then enable motor6 once, verify no initial jump, write the single +8count goal once. Never restore stale goal0.
- Monitor for up to2 seconds; stop on overshoot, alarm, temperature, restriction/state change or other-axis drift. If not reached, do NOT raise power or retry motion.
- Finally disable ONLY motor6 and read back torque0. Two bounded idempotent disable attempts are allowed. If not confirmed, demand immediate manual power cutoff. This cannot guarantee stopping on communications/power failure.
- Close port. No automatic return and no writes to IDs1..5. Low speed/acceleration/torque RAM settings remain; old settings are logged, not automatically restored. Goal remains the small test target until explicitly aligned for a future operation.

The sampled feedback and software guard are NOT a safety-rated controller, nor proof of collision freedom. User observation and cutoff remain necessary. Keep the empty arm supported; all body axes stay unpowered.

## Validation and sources
95 offline tests pass (14 for this microstep implementation plus previous tests). Covers no hardware by default, TTY gating, no writes before consent, old-goal preload ordering, other-ID/write blocking, fixed goal step, endpoints, scaling, stuck motor without power escalation, overshoot, other-axis drift, uncertain enable and disable-failure warning. No test implies actual movement succeeded.

Manufacturer SDK confirms position/speed/acceleration packing and time0:
- https://github.com/ftservo/FTServo_Arduino/blob/main/src/SMS_STS.cpp (WritePosEx)
- https://github.com/ftservo/FTServo_Arduino/blob/main/src/SMS_STS.h (register addresses40..49)
Local rustypot v1.13.1 definitions were cross-checked. Online summaries sometimes confuse decimal48 with hex0x48; code uses named definitions, not that erroneous hex interpretation. Physical scaling of firmware extension registers remains unverified; do not claim a measured speed/torque ceiling.

## Current launch state
Opened one Terminal for the test. Initial process PID68401. Journal motion-tests/motor6-20261031T153133-432245Z.jsonl reached waiting_move6_confirmation, writes_armed=false and0 write attempts at inspection. Do not launch a duplicate. Actual result must be read from the journal and independently confirmed before calling this test completed.
