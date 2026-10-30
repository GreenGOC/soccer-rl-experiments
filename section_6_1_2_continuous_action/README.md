# Section 6.1.2 — Continuous Actions

Trains a single learned agent with DDPG in a custom soccer environment containing scripted teammates and opponents. Observations represent variable-sized entity sets; `gee_encoder.py` performs masked entity pooling.

## Run

```powershell
python section_6_1_2_continuous_action/train_continuous.py --algorithm ddpg --modes 2v2 --episodes 10 --seeds 0 --warmup-steps 20
```

This reduced run is an execution check. The default DDPG workload spans seven modes, two seeds, and 8,000 episodes per mode/seed (112,000 episodes). **SAC is not implemented**, despite being listed as a CLI choice; use `--algorithm ddpg`. Results are saved under this section’s `results` directory.

## Files

- `environment_continuous_action.py` — continuous-action environment
- `gee_encoder.py` — masked variable-entity encoder
- `ddpg.py` — DDPG agent
- `train_continuous.py` — training entry point
- `plot_results.py` — result plots
