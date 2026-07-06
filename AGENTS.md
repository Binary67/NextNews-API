# AGENTS.md

## Project Context

- This project is still in active development.
- Database migrations are not needed at this stage.
- Do not create, edit, or run database migration files unless explicitly requested.
- For schema changes, update the current schema/model definitions directly.
- Do not add legacy database compatibility paths for old schemas.
- Use the codebase virtual environment for Python commands, tests, and scripts. Prefer
  `./.venv/bin/python` and `./.venv/bin/pytest` over system Python or global tools.
