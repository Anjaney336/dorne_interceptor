# STPA-lite Notes

## Control Actions
- CA-1: Apply EKF update from spoofed measurement.
- CA-2: Send interceptor velocity command.
- CA-3: Publish telemetry to operator dashboard.
- CA-4: Publish release artifacts.

## Unsafe Control Actions
- UCA-1: CA-1 applied without spoofing trust down-scaling.
- UCA-2: CA-2 exceeds speed/acceleration constraints.
- UCA-3: CA-3 omits critical threat or state fields.
- UCA-4: CA-4 publishes stale/unvalidated artifacts.

## Safety Constraints
- SC-1: EKF must apply innovation-based trust scaling.
- SC-2: Controller must enforce constraint envelope.
- SC-3: Telemetry schema must preserve required fields.
- SC-4: Release must pass manifest + budget + traceability gates.
