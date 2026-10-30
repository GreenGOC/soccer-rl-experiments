"""Train continuous-action agents (DDPG/SAC) for Section 6-1-2.

Usage:
    python train_continuous.py --algorithm ddpg --modes all --episodes 8000 --seeds 0,1,2,3,4
    python train_continuous.py --algorithm sac --modes all --episodes 8000 --seeds 0,1,2,3,4
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
from environment_continuous_action import ContinuousActionSoccerEnv
from ddpg import DDPGAgent, ReplayBuffer as DDPGReplay


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--algorithm", type=str, default="ddpg", choices=["ddpg", "sac"])
    p.add_argument("--modes", type=str, default="all", help="Comma-separated modes or 'all'")
    p.add_argument("--episodes", type=int, default=8000)
    p.add_argument("--max-steps", type=int, default=250)
    p.add_argument("--seeds", type=str, default="0,1")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--warmup-steps", type=int, default=2000)
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
    out_csv = out_dir / f"{args.algorithm}_results.csv"

    cols = [
        "algorithm",
        "mode",
        "seed",
        "episode",
        "timestep",
        "episode_reward",
        "success",
        "episode_length",
        "mean_actor_loss",
        "mean_value_loss",
        "pass_success",
    ]

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()

        for mode in modes:
            _, team_size = parse_mode(mode)
            for seed in seeds:
                np.random.seed(seed)
                torch.manual_seed(seed)

                env = ContinuousActionSoccerEnv(
                    max_steps=args.max_steps,
                    seed=seed,
                    max_teammates=team_size - 1,
                    max_opponents=team_size,
                    randomize_counts=False,
                )

                if args.algorithm == "ddpg":
                    agent = DDPGAgent(
                        ego_dim=env.ego_dim,
                        entity_dim=env.entity_dim,
                        max_teammates=env.max_teammates,
                        max_opponents=env.max_opponents,
                        action_dim=env.action_dim,
                        device=args.device,
                    )
                    replay = DDPGReplay()

                total_steps = 0

                for ep in range(args.episodes):
                    obs = env.reset(seed=seed * 100_000 + ep)
                    done = False
                    ep_reward = 0.0
                    ep_len = 0
                    success = 0
                    pass_count = 0.0
                    actor_losses = []
                    value_losses = []

                    while not done and ep_len < args.max_steps:
                        if total_steps < args.warmup_steps:
                            action = np.random.uniform(-1.0, 1.0, size=(env.action_dim,)).astype(np.float32)
                        else:
                            if args.algorithm == "ddpg":
                                action = agent.select_action(obs, noise_std=0.15, evaluate=False)
                            else:
                                action = agent.select_action(obs, evaluate=False)

                        step = env.step(action)
                        replay.add(obs, action, step.reward, step.obs, step.done)

                        obs = step.obs
                        ep_reward += step.reward
                        ep_len += 1
                        total_steps += 1
                        pass_count += step.info.get("pass_success", 0.0)

                        if total_steps >= args.warmup_steps:
                            losses = agent.update(replay, batch_size=args.batch_size)
                            if args.algorithm == "ddpg":
                                actor_losses.append(losses["actor_loss"])
                                value_losses.append(losses["critic_loss"])
                            else:
                                actor_losses.append(losses["actor_loss"])
                                value_losses.append(losses["q_loss"])

                        if step.info.get("scored", 0.0) > 0.5:
                            success = 1
                        done = step.done

                    writer.writerow(
                        {
                            "algorithm": args.algorithm,
                            "mode": mode,
                            "seed": seed,
                            "episode": ep,
                            "timestep": total_steps,
                            "episode_reward": float(ep_reward),
                            "success": success,
                            "episode_length": ep_len,
                            "mean_actor_loss": float(np.mean(actor_losses) if actor_losses else 0.0),
                            "mean_value_loss": float(np.mean(value_losses) if value_losses else 0.0),
                            "pass_success": float(pass_count),
                        }
                    )

                    if (ep + 1) % 100 == 0:
                        print(
                            f"[{args.algorithm.upper()}] mode={mode} seed={seed} ep={ep+1}/{args.episodes} "
                            f"reward={ep_reward:.2f} success={success}"
                        )

    print(f"Saved {out_csv} for modes={','.join(modes)}")


if __name__ == "__main__":
    main()
