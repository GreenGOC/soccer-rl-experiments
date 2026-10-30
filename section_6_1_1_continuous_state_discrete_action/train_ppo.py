"""Train PPO (discrete) for Section 6-1-1.

Usage:
    python train_ppo.py --modes all --episodes 8000 --seeds 0,1,2,3,4
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from soccer_modes import SUPPORTED_MODES, parse_mode
from environment_continuous_state import ContinuousStateDiscreteActionSoccerEnv
from ppo_discrete import PPOAgentDiscrete, RolloutBuffer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--modes", type=str, default="all", help="Comma-separated modes or 'all'")
    p.add_argument("--episodes", type=int, default=8000)
    p.add_argument("--max-steps", type=int, default=250)
    p.add_argument("--seeds", type=str, default="0,1")
    p.add_argument("--rollout-steps", type=int, default=2000)
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
        mode, _ = parse_mode(m)
        out.append(mode)
    if not out:
        raise ValueError("No valid modes provided.")
    return out


def main() -> None:
    args = parse_args()
    modes = resolve_modes(args.modes)
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "ppo_results.csv"

    cols = [
        "algorithm",
        "mode",
        "seed",
        "episode",
        "timestep",
        "episode_reward",
        "success",
        "episode_length",
        "mean_loss",
    ]

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()

        for mode in modes:
            _, team_size = parse_mode(mode)
            for seed in seeds:
                np.random.seed(seed)
                torch.manual_seed(seed)

                env = ContinuousStateDiscreteActionSoccerEnv(team_size=team_size, max_steps=args.max_steps, seed=seed)
                agent = PPOAgentDiscrete(obs_dim=env.obs_dim, action_dim=env.action_n, device=args.device)

                buffer = RolloutBuffer()
                total_steps = 0

                for ep in range(args.episodes):
                    obs = env.reset(seed=seed * 100_000 + ep)
                    done = False
                    ep_reward = 0.0
                    ep_len = 0
                    success = 0
                    losses = []

                    while not done and ep_len < args.max_steps:
                        action, logp, value = agent.act(obs)
                        step = env.step(action)

                        buffer.add(obs, action, logp, step.reward, float(step.done), value)
                        obs = step.obs

                        ep_reward += step.reward
                        ep_len += 1
                        total_steps += 1

                        if step.info.get("scored", 0.0) > 0.5:
                            success = 1

                        if len(buffer) >= args.rollout_steps:
                            with torch.no_grad():
                                x = torch.as_tensor(obs, dtype=torch.float32, device=agent.device).unsqueeze(0)
                                _, v = agent.model(x)
                                last_v = float(v.item()) if not step.done else 0.0
                            losses.append(agent.update(buffer, last_value=last_v))
                            buffer.clear()

                        done = step.done

                    writer.writerow(
                        {
                            "algorithm": "ppo_discrete",
                            "mode": mode,
                            "seed": seed,
                            "episode": ep,
                            "timestep": total_steps,
                            "episode_reward": float(ep_reward),
                            "success": success,
                            "episode_length": ep_len,
                            "mean_loss": float(np.mean(losses) if losses else 0.0),
                        }
                    )

                    if (ep + 1) % 100 == 0:
                        print(
                            f"[PPO] mode={mode} seed={seed} ep={ep+1}/{args.episodes} "
                            f"reward={ep_reward:.2f}"
                        )

                if len(buffer) > 0:
                    agent.update(buffer, last_value=0.0)
                    buffer.clear()

    print(f"Saved {out_csv} for modes={','.join(modes)}")


if __name__ == "__main__":
    main()
