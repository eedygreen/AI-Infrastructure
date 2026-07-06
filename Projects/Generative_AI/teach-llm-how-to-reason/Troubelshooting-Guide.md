# Troubleshooting commands

### Create Virtual Environments
uv python pin `${version_number}`: Pins the project to use Python version_number, e.g 3.12.3 by creating a .python-version file. Ensures consistent Python version across environments. <br>

`uv python pin 3.12.3`

### uv init
Initializes a new Python project, creating a pyproject.toml file and setting up the project structure.<br>

`uv init`
### uv sync 
Creates a virtual environment (if needed) and installs dependencies based on  pyproject.toml or a lockfile, ensuring a reproducible environment. <br>

`uv sync`
### uv add ipykernel pip 
Adds ipykernel (for Jupyter integration) and pip to the project dependencies. <br>
`uv add ipykernel pip`

### uv pip install -r requirements.txt
Installs packages listed in requirements.txt using uv's pip-compatible interface. <br>
`uv pip install -r requirements.txt`

### Sync changes in Requirements.txt file
To release the uv lock or cache and apply changes in requirements.txt,

`uv pip sync --refresh requirements.txt`

This forces uv to re-resolve dependencies and update the environment based on the new requirements.

Alternatively <br>
`uv cache clean`   