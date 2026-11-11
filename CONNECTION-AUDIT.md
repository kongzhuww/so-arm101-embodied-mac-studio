# Connection side-effect audit (static only)

Scope: pinned LeRobot 3f797de5952432284a7c77ad2760218cb74e2f26, installed rustypot 1.13.1 Python API declarations, and upstream rustypot v1.13.1 source tag at commit 7aa85e13143594ea4fd4ed9f29e3e29d39b5de85. Source-only checkout at audit/rustypot-v1.13.1; not built or installed. No serial port opened, no hardware I/O performed. User confirmed IDs 1..6 in base-to-gripper order and that 12V remains disconnected.

## Unsafe for an initial read-only inspection

`src/lerobot/robots/so_follower/so_follower.py`:
- `SOFollower.connect()` (line 111): opens bus, may run calibration by default, then always calls configure().
- Passing calibrate=False only skips the calibration branch, NOT configure().
- `configure()` (line 145): uses torque_disabled(), configures motor settings, operating mode and PID; also writes gripper limits.
- `calibrate()` (line 134) records and writes calibration; not a read-only operation.
- `disconnect()` (line 216): by default requests torque disable through config.

`src/lerobot/motors/motors_bus.py`:
- `configure_motors()` writes return delay, acceleration and possibly Phase bit 4.
- `torque_disabled()` disables torque and ALWAYS enables torque on exit, even on exceptions. Not appropriate for read-only tests.
- `disconnect(disable_torque=True)` is the default: writes torque state before closing. For a read-only bus inspection, explicitly use disable_torque=False.

## Narrow candidate path (not executed)

- Construct SerialMotorsBus directly, not SOFollower.
- Establish user-confirmed expected IDs, not an unbounded scan; official default joint mapping is IDs 1..6 but user mapping is not yet confirmed.
- connect(handshake=False) only opens rustypot.Bus in the inspected Python code, with default baudrate 1,000,000 and timeout 50ms. No calibration or configure call.
- Read allowlisted registers individually with normalize=False and num_retry=0, starting with Model_Number; stop on an unexpected model or transport failure. Do not automatically change baudrate, IDs or firmware.
- Optional raw readings, only after model validation: Present_Position, Torque_Enable, Present_Voltage, Present_Temperature. Voltage units must be checked against the model definition before displaying physical units.
- Always close with disconnect(disable_torque=False), including exception paths.
- read() -> _read() -> rustypot.Bus.read_register_with_error() in inspected Python layer.

## Audit limits and hardware precautions

Upstream version-matched source review:
- src/bus.rs:359-371: PyBus::new opens serialport with specified timeout, then creates a Bus. Bus::new at line 40 only stores port/model definitions and creates protocol handlers, no motor commands.
- src/bus.rs:375-376: close drops the underlying Bus/port; no explicit torque/configuration write.
- src/bus.rs:429: read_register_with_error invokes only a retried read operation; no automatic initialization or torque setting.
- src/servo/definition.rs:78-88 resolves the named register, calls read_with_error and decodes response.
- src/dynamixel_protocol/mod.rs:677-687 sends one read packet then reads status.
- src/dynamixel_protocol/v1.rs:50-55 creates a Read instruction (0x02, enum mapping at line 268), distinct from Write. Parameters are address and read length, not a goal value.
- src/servo/feetech/sts3215.rs: model_number address 3 length 2, torque_enable address 40 length 1, present_position address 56 length 2. Reading torque_enable does not change it. Motor resolution is 4096; raw position is not an uncalibrated joint angle.

Conclusion: the selected open/read/close path in version-matched source has no motor register writes, homing, calibration or explicit torque enable/disable. This is static evidence, not an absolute hardware guarantee: the installed compiled wheel was not reproducibly rebuilt/bitwise matched to the Git tag, and the transitive serialport OS-driver implementation/control-line behavior was not fully audited.

A read-only protocol request still transmits bytes; it is not passive USB enumeration. Serial open may affect adapter control lines, and power-up may restore an existing servo state. User must confirm supply state and remain present, with the arm safely supported and workspace clear. No software read-only approach guarantees absence of power-up movement.

Next step: prepare and offline-test a fail-closed read-only script before any power-up. Expected IDs confirmed as 1..6; first read model identity, then torque state and raw position. Stop on timeouts, unexpected model or alarm. Do not turn torque off as an automatic cleanup operation (a supported arm could fall). Hardware use still requires separate explicit authorization and a supported/cleared arm with accessible power cutoff. No firmware updates, motor setup, calibration, or torque changes are authorized by this audit.
