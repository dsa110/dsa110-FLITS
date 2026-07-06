# FLITS Scripts

Utility scripts for demos, repo tooling, and data-migration audits.

## Structure

```
scripts/
├── demos/
│   └── run_single_burst.py   # single-burst analysis end-to-end
├── hpcc/                     # cluster foreground-search launch scripts
├── manuscript/               # manuscript figure regeneration
├── migration/                # host-migration audit scripts (machine_inventory.yaml)
├── entire_checkpoint.py      # post-commit Entire tracing checkpoint hook
└── query_machine_inventory.py # query machine_inventory.yaml
```

## Demos

### `run_single_burst.py`

Run a single burst analysis end-to-end (any burst/telescope):

```bash
python scripts/demos/run_single_burst.py --burst casey --telescope dsa
```

## See Also

- Simulation scripts: `simulation/scripts/`
- Main pipelines: `scattering/`, `scintillation/`
