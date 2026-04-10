# GitHub Project Board Setup (Project v2)

Use this once per repository to create a professional delivery board.

## Recommended Board
Name: `Dorne Interceptor Delivery`

Fields:
- `Status` (Single select): `Backlog`, `Ready`, `In Progress`, `Review`, `Done`
- `Priority` (Single select): `P0`, `P1`, `P2`
- `Area` (Single select): `Perception`, `Tracking`, `Navigation`, `Simulation`, `Backend`, `CI/CD`, `Docs`
- `Risk` (Single select): `Low`, `Medium`, `High`

Views:
- `Board` grouped by `Status`
- `Table` sorted by `Priority`
- `By Area` grouped by `Area`

Automation suggestions:
- PR opened -> set status `Review`
- PR merged -> set status `Done`
- New issue with label `bug` -> priority `P1`

## Manual Setup Steps
1. Open GitHub -> your repo -> `Projects` -> `New project`.
2. Choose `Board`, name it `Dorne Interceptor Delivery`.
3. Add the fields and options above.
4. Set up saved views.
5. Add this project link to the repository README.
