# Hyperparameter Tables

| Algorithm | Parameter | Value |
|---|---|---|
| q_learning | alpha | 0.1 |
| q_learning | gamma | 0.1/0.5/0.89 |
| q_learning | epsilon | 1.0 -> 0.05 linear |
| sarsa | alpha | 0.1 |
| sarsa | gamma | 0.1/0.5/0.89 |
| sarsa | epsilon | 1.0 -> 0.05 linear |
| actor_critic_tabular | alpha_v | 0.1 |
| actor_critic_tabular | alpha_pi | 0.05 |
| actor_critic_tabular | gamma | 0.1/0.5/0.89 |
| iql | alpha | 0.1 |
| iql | gamma | 0.89 |
| jal | alpha | 0.08 |
| jal | gamma | 0.89 |
| dqn | learning_rate | 3e-4 |
| dqn | batch_size | 128 |
| dqn | gamma | 0.99 |
| ppo_discrete | learning_rate | 3e-4 |
| ppo_discrete | clip_ratio | 0.2 |
| ppo_discrete | gae_lambda | 0.95 |
| ddpg | lr_actor/lr_critic | 3e-4 / 3e-4 |
| ddpg | gamma | 0.99 |
| ddpg | tau | 0.01 |
| sac | learning_rate | 3e-4 |
| sac | gamma | 0.99 |
| sac | alpha | 0.2 |
| ctde_mappo | lr_actor/lr_critic | 3e-4 / 3e-4 |
| ctde_mappo | clip_ratio | 0.2 |
| ctde_mappo | gae_lambda | 0.95 |
| ippo | learning_rate | 3e-4 |
| ippo | clip_ratio | 0.2 |
| ippo | gae_lambda | 0.95 |