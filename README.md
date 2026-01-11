# Tox21 GNN Project

This project uses Graph Neural Networks to predict features of the Tox21 dataset.

## Setup and usage

I added the possibility to build the virtual environment, install the needed
dependencies and run the scripts using `uv`.
Assuming `uv` is installed, to install the necessary dependencies, run this command in the project folder:
```bash
uv sync
```

To run a script using the configured environment (ensuring GPU libraries are loaded):
```bash
uv run --env-file .env python <script_name>.py
```

