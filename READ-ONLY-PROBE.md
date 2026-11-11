# Offline-tested read-only diagnostic

## Current state
12V is still disconnected according to the user. No real serial port was opened during development or tests. Do not power the arm solely because these tests passed. Power-up behavior is unverified and cannot be controlled by a software read-only gate.

## Files
- scripts/read_only_probe.py
- scripts/test_read_only_probe.py

## Safe offline commands
From /Users/kongzhu/so-arm101-act:

```bash
bash run python -B scripts/read_only_probe.py
bash run python -B -m unittest discover -s scripts -p 'test_read_only_probe.py' -v
```

The first command only displays the plan. The second uses fake/mocked serial transports. Both commands have been run; 23 tests passed.

## Design
The hardware path deliberately uses a minimal pyserial protocol adapter rather than an SDK robot connection. The code at the sole transmit boundary accepts only 18 exact packets: 6 unicast IDs x 3 read registers. No writes, broadcast scans, setup, homing, retries, ID changes or baudrate searches. The fixed baudrate is 1Mbps, the fixed device path is /dev/cu.usbserial-130, and the enumerated USB VID/PID must match 1a86:7523. Matching a USB converter VID/PID is not unique physical-board authentication.

Register definitions checked against rustypot v1.13.1 STS3215:
- Model_Number: address 3, 2 bytes, expected numeric model 777.
- Torque_Enable: address 40, 1 byte.
- Present_Position: address 56, 2 bytes, signed-magnitude bit 15. Not a calibrated joint angle.

All six model identities are checked first. Then each motor's torque and position are read; any nonzero torque state stops the probe without changing it. A valid model number does not verify gearbox ratio, voltage rating, assembly, calibration, mechanical safety or ID-to-joint mapping.

Responses must have exact header, expected ID and length, correct checksum, and zero status/alarm. Short reads, bad responses, partial transmission and timeouts abort with no retries. On exception or Ctrl+C, only the port is closed; no automatic torque-disable command is sent (that could let an arm fall).

Hardware mode is gated behind BOTH --execute and --ack-power-risk. These options were tested only against mocks; no real hardware invocation occurred. Do not invoke hardware mode until separately authorized after supporting the arm, clearing its workspace, confirming power connection and keeping a cutoff accessible. Do not hold hands between joints or force unpowered joints into a new pose. Inactive RTS/DTR are requested before port open, but OS/adapter control-line transients cannot be ruled out.

## Test coverage
23 tests include exact known READ bytes, all single-byte packet mutations, block write/broadcast/other registers, no hardware in default mode, both flags required, normal sequence, model mismatch, nonzero torque, signed position, partial transmit, timeout/truncation, deadline, malformed header/ID/length, checksum, alarm, success/failure/interrupt cleanup, missing/wrong USB identity, mocked serial open configuration and open failure cleanup.

## Next step
Review physical support and power cutoff with the user BEFORE first energizing. Actual communication and power-up movement remain unverified. This script is diagnostic, not an emergency stop or a safety-rated controller.
