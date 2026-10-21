# Archived THOR configs

Stale configuration files from earlier experiments, kept for provenance.
**None of these match the current `train_thor.py`.** The live reference is
`../Reference_config.yaml`.

They predate several changes to the config schema — among them the
`LR_Schedule` block, `Fractional_training`, the PID settings
(`PID_P` / `PID_I` / `PID_D`, `lambda_self`), `concat_modalities_*`, and the
`records_type: 'ESM'` data format. Expect a run started from one of these to
fail on a missing key.

| | |
|---|---|
| `Reference_config.yaml` | The previous reference config. Superseded by the file that now carries that name. |
| `config_home.yaml`, `config_start.yaml` | Early working configs. |
| `Experiment_22_10_07_*.yaml` | Per-COG experiments from Oct 2022 — paired `_psychro` / `_thermal` runs over COG0039, COG2032, KOG0441 and the 11137 / 11511 families. |
| `EE_22_09_27/`, `EE_22_10_10/` | Experiment sets from Sept/Oct 2022. |
| `CuSOD/`, `PETAses/` | Family-specific configs for Cu/Zn superoxide dismutase and PETases. |
| `HyperParameterOptimization/` | Sweep configs. |

The configs for the ET regressor are **not** archived — they are still in use,
in `../../Classifier/`.
