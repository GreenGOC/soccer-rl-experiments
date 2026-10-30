# Section 3.1 — Tabular TD Methods

Compares tabular Q-learning and SARSA in a discrete, single-agent soccer environment with six actions and a scripted opponent. The learner state includes its position, the ball position, and possession. `sensitivity_analysis.py` runs both methods over six discount factors and two seeds; `plot_results.py` produces learning/outcome plots.

## Run

From the repository root:

```powershell
python section_3_1_td_methods/sensitivity_analysis.py
python section_3_1_td_methods/plot_results.py
```

The sensitivity script has no CLI flags. Its default 192,000 episodes can require substantial compute; adjust the values in `config.py` before a small test. Run it from the repository root because its result paths are relative to the current directory. The trainers also write `q_learning_log.txt` and `sarsa_log.txt` in the current directory; these are overwritten on later runs.

## Files

- `environment_discrete.py` — single-agent environment
- `q_learning.py`, `sarsa.py` — learning algorithms
- `sensitivity_analysis.py` — multi-run experiment
- `plot_results.py` — learning curves and outcome summaries
