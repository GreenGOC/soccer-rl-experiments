# Section 7.1 — Deep Multi-Agent RL

Compares Independent PPO (IPPO) with a MAPPO-style method using centralized training and decentralized execution (CTDE) with a centralized critic. Both trainers use a rule-based hierarchical skill adapter and a pool of mirrored policy snapshots for opponents.

## Run

```powershell
python section_7_1_deep_marl/train_independent.py --modes 2v2 --episodes 10 --seeds 0
python section_7_1_deep_marl/train_ctde.py --modes 2v2 --episodes 10 --seeds 0
python section_7_1_deep_marl/compare_ctde_vs_independent.py
```

The reduced runs are for execution checks. Each trainer defaults to seven modes, two seeds, and 8,000 episodes per mode/seed (112,000 episodes), plus PPO updates and opponent snapshots. Results are saved under this section’s `results` directory.

## Files

- `environment_continuous_action.py` — multi-agent continuous-control task
- `ippo.py`, `ctde_mappo.py` — policy-learning methods
- `hierarchical_skills.py`, `gee_encoder.py` — skill and entity representations
- `train_independent.py`, `train_ctde.py` — training entry points
- `compare_ctde_vs_independent.py` — comparison analysis
