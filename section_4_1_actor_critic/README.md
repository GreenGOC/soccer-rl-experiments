# Section 4.1 — Tabular Actor–Critic

Implements a softmax tabular policy (actor) and state-value critic, updating the policy with the temporal-difference error and optionally including an entropy term. Experiments reuse the discrete single-agent soccer task and vary discount factors and random seeds.

## Run

```powershell
python section_4_1_actor_critic/run_experiments.py
python section_4_1_actor_critic/plot_variance.py
```

The default experiment runs 96,000 episodes. The runner has no CLI flags; change settings in `config.py` for a smaller test. Outputs are saved under this section’s `results` directory.

## Files

- `environment_discrete.py` — discrete single-agent task
- `actor_critic_tabular.py` — tabular policy and value updates
- `run_experiments.py` — training across configured discounts and seeds
- `plot_variance.py` — learning, variance, and outcome comparisons
