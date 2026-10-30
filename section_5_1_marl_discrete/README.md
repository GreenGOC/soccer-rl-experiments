# Section 5.1 — Discrete Multi-Agent RL

Explores cooperative and competitive team rewards in a custom discrete soccer environment. The experiment runners compare Distributed Q-Learning, Friend-or-Foe Q, Minimax-Q, FMQ, and Joint-Action Learning against heuristic-controlled players. An Independent Q-Learning implementation is also present, but is not called by these two runners.

## Run

```powershell
python section_5_1_marl_discrete/cooperative_experiment.py --mode 2v2 --episodes-dql 100 --episodes-fof 100 --episodes-minimax 100 --episodes-fmq 100 --episodes-jal 100
python section_5_1_marl_discrete/competitive_experiment.py --mode 2v2 --episodes-dql 100 --episodes-fof 100 --episodes-minimax 100 --episodes-fmq 100 --episodes-jal 100
python section_5_1_marl_discrete/plot_results.py
```

The reduced commands are for execution checks, not conclusions. Defaults run five learners over two modes and two seeds (160,000 episodes per runner). This section supports 2v2 and 3v3; do not assume the broader modes in `soccer_modes.py` apply here. Outputs are written to this section’s `results` directory.

## Files

- `environment_discrete_multi.py` — multi-agent environment and rewards
- `advanced_q_algorithms.py`, `joint_action_learners.py` — multi-agent learners
- `independent_q_learning.py` — independent learner implementation
- `cooperative_experiment.py`, `competitive_experiment.py` — experiment entry points
- `analyze_non_stationarity.py`, `plot_results.py` — analysis and plots
