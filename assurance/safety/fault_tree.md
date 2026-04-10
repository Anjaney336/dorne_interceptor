# Fault Tree (Top Event)

Top Event: Unsafe or unverifiable mission behavior reaches release branch.

- OR: Navigation resilience failure
  - EKF spoof gate regression
  - Missing anti-spoof tests
- OR: Control envelope violation
  - Constraint logic regression
  - Missing constraint tests
- OR: Release assurance failure
  - Manifest mismatch
  - Performance budget breach
  - Missing CI gate execution

Mitigation: enforce traceability + budget + manifest gates in CI and pre-release workflow.
