# Section 8.1 — Evaluation

Runs heuristic baselines, aggregates experiment logs, produces plots, and creates hyperparameter tables.

## Run

```powershell
python section_8_1_evaluation/run_all_evaluation.py
```

This script has no CLI flags and is a substantial run: the configured baselines total up to roughly 80 million environment steps. Review `config.py` before running.

### Important result-file caveat

The aggregation configuration expects some per-mode filenames, while current Section 5, 6.1.2, and 7 trainers write combined result files. As a result, aggregation may miss newly generated data and read older per-mode CSVs. Check the source files and timestamps before relying on any aggregate or plot.

## Files

- `heuristic_baseline.py` — baseline policies and episode logs
- `aggregate_metrics.py` — result aggregation
- `plot_evaluation.py` — evaluation plots
- `hyperparameter_tables.py` — parameter tables
- `run_all_evaluation.py` — end-to-end entry point
