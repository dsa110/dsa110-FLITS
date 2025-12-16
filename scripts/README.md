# FLITS Scripts

Utility scripts for development, demos, and testing.

## Structure

```
scripts/
├── demos/              # Demo scripts showing pipeline usage
│   ├── demo_batch.py
│   └── run_single_burst.py
└── dev/                # Development utilities
    └── create_dummy_db.py
```

## Demos

### `demo_batch.py`

Demonstrates batch processing of multiple bursts using the `flits-batch` CLI.

**Usage:**

```bash
python scripts/demos/demo_batch.py
```

### `run_single_burst.py`

Example script for running a single burst analysis end-to-end.

**Usage:**

```bash
python scripts/demos/run_single_burst.py --burst casey --telescope dsa
```

## Development Tools

### `create_dummy_db.py`

Creates a dummy `flits_results.db` for testing database interactions.

**Usage:**

```bash
python scripts/dev/create_dummy_db.py
```

## See Also

- Simulation scripts: `simulation/scripts/`
- Main pipelines: `scattering/`, `scintillation/`
- Analysis notebooks: `analyses/`
