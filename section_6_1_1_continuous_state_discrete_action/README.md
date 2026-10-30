# Section 6.1.1 — Continuous State, Discrete Actions

Compares a replay-buffer DQN with target network against discrete-action PPO. The custom environment uses continuous coordinates, six discrete actions, and a scripted opponent.

## Run

Small execution checks:

```powershell
python section_6_1_1_continuous_state_discrete_action/train_dqn.py --modes 2v2 --episodes 10 --seeds 0 --warmup-steps 20
python section_6_1_1_continuous_state_discrete_action/train_ppo.py --modes 2v2 --episodes 10 --seeds 0
```

For the configured full runs, omit the reduced `--modes`, `--episodes`, and `--seeds` values. Defaults cover seven modes, two seeds, and 8,000 episodes per mode/seed (112,000 episodes per algorithm). Results are written to this section’s `results` directory.

## Files

- `environment_continuous_state.py` — continuous-coordinate task
- `dqn.py`, `ppo_discrete.py` — algorithm implementations
- `train_dqn.py`, `train_ppo.py` — training entry points
- `compare_sample_efficiency.py`, `plot_comparison.py` — analysis and plots
