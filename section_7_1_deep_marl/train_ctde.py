"""Train CTDE MAPPO for Section 7-1.

Usage:
    python train_ctde.py --modes all --episodes 8000 --seeds 0,1,2,3,4
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from config import RESULT_COLUMNS, SUPPORTED_MODES, parse_mode
from environment_continuous_action import MultiAgentContinuousSoccerEnv
from ctde_mappo import Actor, CTDEMAPPO, CTDERollout
from hierarchical_skills import apply_hierarchical_controller


MIRROR_X_IDX = (0, 2, 4, 6, 8, 10)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--modes", type=str, default="all", help="Comma-separated modes or 'all'")
    p.add_argument("--episodes", type=int, default=8000)
    p.add_argument("--max-steps", type=int, default=250)
    p.add_argument("--seeds", type=str, default="0,1")
    p.add_argument("--pool-add-every", type=int, default=25)
    p.add_argument("--pool-max-size", type=int, default=40)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def resolve_modes(spec: str) -> list[str]:
    s = spec.strip().lower()
    if s == "all":
        return list(SUPPORTED_MODES)
    out: list[str] = []
    for raw in spec.split(","):
        m = raw.strip()
        if not m:
            continue
        team_size = parse_mode(m)
        out.append(f"{team_size}v{team_size}")
    if not out:
        raise ValueError("No valid modes provided.")
    return out


def _snapshot_state_dict(module: torch.nn.Module):
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


def _mirror_obs(obs: np.ndarray) -> np.ndarray:
    out = obs.astype(np.float32).copy()
    for idx in MIRROR_X_IDX:
        out[idx] *= -1.0
    return out


def _sample_opponent_actions(
    actor: Actor,
    obs_dict: dict[str, np.ndarray],
    agent_ids: list[str],
    device: torch.device,
) -> dict[str, np.ndarray]:
    if not agent_ids:
        return {}
    obs_arr = np.stack([_mirror_obs(obs_dict[aid]) for aid in agent_ids], axis=0)
    obs_t = torch.as_tensor(obs_arr, dtype=torch.float32, device=device)
    with torch.no_grad():
        mean, log_std = actor(obs_t)
        std = torch.exp(log_std)
        pre_tanh = mean + std * torch.randn_like(mean)
        action = torch.tanh(pre_tanh)
    a_np = action.cpu().numpy().astype(np.float32)
    a_np[:, 0] *= -1.0  # un-mirror x command back to team-1 frame
    residual = {aid: a_np[i] for i, aid in enumerate(agent_ids)}
    return {aid: apply_hierarchical_controller(obs_dict[aid], residual[aid]) for aid in agent_ids}


def main() -> None:
    args = parse_args()
    modes = resolve_modes(args.modes)
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "ctde_results.csv"

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
        writer.writeheader()

        for mode in modes:
            team_size = parse_mode(mode)
            for seed in seeds:
                np.random.seed(seed)
                torch.manual_seed(seed)
                opponent_rng = np.random.default_rng(seed + 1_111)

                env = MultiAgentContinuousSoccerEnv(team_size=team_size, max_steps=args.max_steps, seed=seed)
                agent = CTDEMAPPO(
                    obs_dim=env.local_obs_dim,
                    state_dim=env.global_state_dim,
                    action_dim=env.action_dim,
                    team_size=env.team_size,
                    device=args.device,
                )

                opponent_actor = Actor(env.local_obs_dim, env.action_dim).to(agent.device)
                opponent_pool = [_snapshot_state_dict(agent.actor)]

                total_steps = 0
                for ep in range(args.episodes):
                    obs, state = env.reset(seed=seed * 100_000 + ep)
                    done = False
                    ep_team0_reward = 0.0
                    ep_team1_reward = 0.0
                    ep_len = 0
                    ep_pass = 0.0
                    success = 0

                    pool_idx = int(opponent_rng.integers(0, len(opponent_pool)))
                    opponent_actor.load_state_dict(opponent_pool[pool_idx])
                    opponent_actor.eval()

                    rollout = CTDERollout([], [], [], [], [], [], [])

                    while not done and ep_len < args.max_steps:
                        residual_actions, logps, value, critic_input = agent.act(obs, state, env.controlled_ids)
                        actions = {
                            aid: apply_hierarchical_controller(obs[aid], residual_actions[aid]) for aid in env.controlled_ids
                        }
                        opp_obs = env.get_team_obs(1)
                        opp_actions = _sample_opponent_actions(opponent_actor, opp_obs, env.team1_ids, agent.device)
                        out = env.step(actions, team1_actions=opp_actions)

                        obs_arr = np.stack([obs[aid] for aid in env.controlled_ids], axis=0)
                        act_arr = np.stack([residual_actions[aid] for aid in env.controlled_ids], axis=0)
                        logp_arr = np.array([logps[aid] for aid in env.controlled_ids], dtype=np.float32)

                        rollout.obs_steps.append(obs_arr)
                        rollout.act_steps.append(act_arr)
                        rollout.logp_steps.append(logp_arr)
                        rollout.state_steps.append(critic_input)
                        rollout.reward_steps.append(float(out.info["team0_reward"]))
                        rollout.done_steps.append(float(out.done))
                        rollout.value_steps.append(float(value))

                        obs = out.local_obs
                        state = out.global_state
                        done = out.done
                        ep_len += 1
                        total_steps += 1

                        ep_team0_reward += float(out.info["team0_reward"])
                        ep_team1_reward += float(out.info["team1_reward"])
                        ep_pass += float(out.info.get("pass_count", 0.0))
                        if out.info.get("success", 0.0) > 0.5:
                            success = 1

                    if done:
                        last_value = 0.0
                    else:
                        with torch.no_grad():
                            c_in = agent.build_critic_input(obs, state, env.controlled_ids)
                            c_t = torch.as_tensor(c_in, dtype=torch.float32, device=agent.device).unsqueeze(0)
                            last_value = float(agent.critic(c_t).item())

                    mean_loss = agent.update(rollout, last_value=last_value)

                    if (ep + 1) % args.pool_add_every == 0:
                        opponent_pool.append(_snapshot_state_dict(agent.actor))
                        if len(opponent_pool) > args.pool_max_size:
                            opponent_pool.pop(0)

                    writer.writerow(
                        {
                            "algorithm": "ctde_mappo",
                            "mode": mode,
                            "seed": seed,
                            "episode": ep,
                            "timestep": total_steps,
                            "team0_reward": ep_team0_reward,
                            "team1_reward": ep_team1_reward,
                            "success": success,
                            "episode_length": ep_len,
                            "pass_count": ep_pass,
                            "mean_loss": mean_loss,
                        }
                    )

                    if (ep + 1) % 100 == 0:
                        print(
                            f"[CTDE] mode={mode} seed={seed} ep={ep+1}/{args.episodes} "
                            f"team0_reward={ep_team0_reward:.2f} success={success} pool={len(opponent_pool)}"
                        )

    print(f"Saved {out_csv} for modes={','.join(modes)}")


if __name__ == "__main__":
    main()
