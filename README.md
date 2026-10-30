# Soccer Reinforcement Learning Experiments

A collection of custom soccer simulators and reinforcement-learning experiments, progressing from tabular single-agent methods to deep multi-agent control. The project compares learning approaches across discrete and continuous state/action representations and includes baseline evaluation and plotting utilities.

> **Scope:** These are educational, simplified simulations—not a Gymnasium benchmark, a robot-control system, or a reproduction of any one paper. The environments and episode logs differ between sections, so results should only be compared within the experimental setup that produced them.

## What this project explores

| Section | Focus | Implementations |
|---|---|---|
| `section_3_1_td_methods` | Tabular temporal-difference learning in a discrete, single-agent soccer task | Q-learning, SARSA, discount-factor sensitivity analysis |
| `section_4_1_actor_critic` | Tabular policy-gradient learning | Softmax Actor–Critic with a value critic and optional entropy term |
| `section_5_1_marl_discrete` | Discrete multi-agent cooperation and competition | Distributed Q-Learning, Friend-or-Foe Q, Minimax-Q, FMQ, and Joint-Action Learning |
| `section_6_1_1_continuous_state_discrete_action` | Continuous observations with discrete controls | DQN with replay/target network, discrete PPO |
| `section_6_1_2_continuous_action` | Continuous control with variable-entity observations | DDPG and a masked entity-pooling encoder |
| `section_7_1_deep_marl` | Deep multi-agent policy learning | IPPO and MAPPO-style centralized training with decentralized execution (CTDE) |
| `section_8_1_evaluation` | Baselines and result analysis | Heuristic policies, metric aggregation, plots, hyperparameter tables |

The simulators model simplified movement, possession, passing/kicking, goals, rewards, and scripted opponents. Some experiments use a single learned player while teammates or opponents follow rules; others train teams. The helper `soccer_modes.py` supports symmetric team-size labels from 2v2 through 13v13, but individual sections support different subsets.

## Repository layout

Each `section_*` directory has its own `README.md`, configuration, training/evaluation scripts, and `results` folder. Some sections also use the root helper `soccer_modes.py`; there is no shared environment API or top-level experiment runner.

Training scripts generally write episode CSV logs and configuration snapshots to their section’s `results` directory. Plotting scripts save PNG figures there. Existing figures are retained as examples; generated CSV and text logs are excluded from Git because some are large and can be regenerated.

## Setup

Python 3.10 or later is recommended.

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The experiments use NumPy, Matplotlib, and PyTorch. PyTorch will use CUDA when available in scripts that expose a `--device` option; CPU is also supported.

## Running experiments

Run commands from the repository root unless a section README says otherwise. Most defaults are **full experiments**, not quick demos: an 8,000-episode run across several modes and seeds can involve tens of millions of environment steps. Actual runtime depends on hardware and episode lengths.

For a small execution check, for example:

```powershell
python section_6_1_1_continuous_state_discrete_action/train_dqn.py --modes 2v2 --episodes 10 --seeds 0 --warmup-steps 20
```

This only checks that a small run can execute; it is not enough data to evaluate or compare a policy. See each section README for its full-run command and controls. The Section 3 sensitivity analysis has no CLI flags and uses a large fixed workload; reduce values in its configuration before using it as a smoke test.

## Results and interpretation

The project records measures such as episode/team reward, scoring or success, episode length, TD error, losses, and pass counts, depending on the experiment. The evaluation section also produces aggregate metrics and plots.

Treat existing plots and logs as exploratory outputs from particular runs, not as evidence that one algorithm is generally superior. Seeds, environment variants, opponent policies, and metrics differ. In particular:

- Section 5 is configured for 2v2 and 3v3, while some other modules expose more modes.
- Section 8’s expected per-mode CSV names do not consistently match the combined filenames written by current Section 5, 6.1.2, and 7 trainers. Verify the input CSVs before interpreting a newly generated aggregate; stale files may otherwise be used.
- Section 8’s full baseline run is substantial (up to roughly 80 million environment steps across its configured baselines).
- Section 6.1.2 exposes `--algorithm sac`, but SAC is not implemented; use `--algorithm ddpg`.

The custom simulators do not model robot dynamics, sensors, communication delays, or sim-to-real transfer, and no real-robot evaluation is included.

## Related reading

These papers provide background on themes explored here; this repository is not a reproduction of their experimental results:

- Li, Z., Bjelonic, F., Klemm, V., & Hutter, M. (2024). [MARLadona—Towards Cooperative Team Play Using Multi-Agent Reinforcement Learning](https://arxiv.org/abs/2409.20326). Background on multi-agent team play, CTDE, and global entity representations.
- Su, Z., Gao, Y., Lukas, E., et al. (2025). [Toward Real-World Cooperative and Competitive Soccer with Quadrupedal Robot Teams](https://arxiv.org/abs/2505.13834). Related work on hierarchical skills and quadrupedal robot soccer; its real-robot setting is distinct from these custom simulations.
- Taourirte, A. & Mia, M. S. (2025). [Multi-Agent Reinforcement Learning and Real-Time Decision-Making in Robotic Soccer for Virtual Environments](https://arxiv.org/abs/2512.03166). A preprint comparing approaches in virtual soccer; reported numbers belong to that paper’s setup and are not results from this repository.

The original workspace also contains course instructions, student reports, source-paper PDFs, and an archive. Those materials are intentionally not included: they are not required to run the code, may have redistribution restrictions, and some contain personal/course information. Cite and redistribute source papers according to their own terms.
