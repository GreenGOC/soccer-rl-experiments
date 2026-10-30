"""Run competitive MARL discrete experiments for Section 5-1.

Usage:
    python competitive_experiment.py            # runs all modes from config into one CSV
    python competitive_experiment.py --mode 2v2
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import csv

from config import (
    EnvConfig,
    DistributedQLearningConfig,
    FriendOrFoeConfig,
    MinimaxQConfig,
    FMQConfig,
    JALConfig,
    ExperimentConfig,
    RESULT_COLUMNS,
    output_dir,
    parse_mode,
    normalize_mode,
    grid_size_for_mode,
)
from environment_discrete_multi import MultiAgentDiscreteSoccerEnv
from advanced_q_algorithms import (
    train_distributed_q_learning,
    train_friend_or_foe_q,
    train_minimax_q_learning,
    train_fmq,
)
from joint_action_learners import train_jal


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", type=str, default="all")
    p.add_argument("--episodes-dql", type=int, default=None)
    p.add_argument("--episodes-fof", type=int, default=None)
    p.add_argument("--episodes-minimax", type=int, default=None)
    p.add_argument("--episodes-fmq", type=int, default=None)
    p.add_argument("--episodes-jal", type=int, default=None)
    return p.parse_args()


def build_env(env_cfg: EnvConfig, seed: int, cooperative: bool) -> MultiAgentDiscreteSoccerEnv:
    return MultiAgentDiscreteSoccerEnv(
        grid_size=env_cfg.grid_size,
        max_steps=env_cfg.max_steps,
        agents_per_team=env_cfg.agents_per_team,
        seed=seed,
        reward_goal=env_cfg.reward_goal,
        reward_concede=env_cfg.reward_concede,
        reward_step=env_cfg.reward_step,
        reward_ball_progress=env_cfg.reward_ball_progress,
        reward_collision=env_cfg.reward_collision,
        reward_out=env_cfg.reward_out,
        cooperative=cooperative,
    )


def run_mode(
    mode: str,
    writer: csv.DictWriter,
    args: argparse.Namespace,
    seeds: tuple[int, ...],
    dql_cfg: DistributedQLearningConfig,
    fof_cfg: FriendOrFoeConfig,
    minimax_cfg: MinimaxQConfig,
    fmq_cfg: FMQConfig,
    jal_cfg: JALConfig,
) -> dict:
    team_size = parse_mode(mode)
    env_cfg = EnvConfig(mode=mode, grid_size=grid_size_for_mode(mode), agents_per_team=team_size)
    for seed in seeds:
        print(f"[DQL competitive] mode={mode} seed={seed}")
        env_dql = build_env(env_cfg, seed, cooperative=False)
        logs_dql = train_distributed_q_learning(
            env=env_dql,
            episodes=args.episodes_dql if args.episodes_dql is not None else dql_cfg.episodes,
            alpha=dql_cfg.alpha,
            gamma=dql_cfg.gamma,
            epsilon_start=dql_cfg.epsilon_start,
            epsilon_end=dql_cfg.epsilon_end,
            epsilon_decay_episodes=dql_cfg.epsilon_decay_episodes,
            consensus_tau=dql_cfg.consensus_tau,
            seed=seed,
            setting="competitive",
            mode=mode,
        )
        for row in logs_dql:
            writer.writerow(row.__dict__)

        print(f"[FoF-Q competitive] mode={mode} seed={seed}")
        env_fof = build_env(env_cfg, seed + 7, cooperative=False)
        logs_fof = train_friend_or_foe_q(
            env=env_fof,
            episodes=args.episodes_fof if args.episodes_fof is not None else fof_cfg.episodes,
            alpha=fof_cfg.alpha,
            gamma=fof_cfg.gamma,
            epsilon_start=fof_cfg.epsilon_start,
            epsilon_end=fof_cfg.epsilon_end,
            epsilon_decay_episodes=fof_cfg.epsilon_decay_episodes,
            seed=seed,
            setting="competitive",
            mode=mode,
        )
        for row in logs_fof:
            writer.writerow(row.__dict__)

        print(f"[Minimax-Q competitive] mode={mode} seed={seed}")
        env_mmq = build_env(env_cfg, seed + 11, cooperative=False)
        logs_mmq = train_minimax_q_learning(
            env=env_mmq,
            episodes=args.episodes_minimax if args.episodes_minimax is not None else minimax_cfg.episodes,
            alpha=minimax_cfg.alpha,
            gamma=minimax_cfg.gamma,
            epsilon_start=minimax_cfg.epsilon_start,
            epsilon_end=minimax_cfg.epsilon_end,
            epsilon_decay_episodes=minimax_cfg.epsilon_decay_episodes,
            seed=seed,
            setting="competitive",
            mode=mode,
        )
        for row in logs_mmq:
            writer.writerow(row.__dict__)

        print(f"[FMQ competitive] mode={mode} seed={seed}")
        env_fmq = build_env(env_cfg, seed + 13, cooperative=False)
        logs_fmq = train_fmq(
            env=env_fmq,
            episodes=args.episodes_fmq if args.episodes_fmq is not None else fmq_cfg.episodes,
            alpha=fmq_cfg.alpha,
            gamma=fmq_cfg.gamma,
            epsilon_start=fmq_cfg.epsilon_start,
            epsilon_end=fmq_cfg.epsilon_end,
            epsilon_decay_episodes=fmq_cfg.epsilon_decay_episodes,
            fmq_c=fmq_cfg.fmq_c,
            seed=seed,
            setting="competitive",
            mode=mode,
        )
        for row in logs_fmq:
            writer.writerow(row.__dict__)

        print(f"[JAL competitive] mode={mode} seed={seed}")
        env_j = build_env(env_cfg, seed + 17, cooperative=False)
        logs_j = train_jal(
            env=env_j,
            episodes=args.episodes_jal if args.episodes_jal is not None else jal_cfg.episodes,
            alpha=jal_cfg.alpha,
            gamma=jal_cfg.gamma,
            epsilon_start=jal_cfg.epsilon_start,
            epsilon_end=jal_cfg.epsilon_end,
            epsilon_decay_episodes=jal_cfg.epsilon_decay_episodes,
            seed=seed,
            setting="competitive",
            mode=mode,
        )
        for row in logs_j:
            writer.writerow(row.__dict__)
    return asdict(env_cfg)


def main() -> None:
    args = parse_args()
    exp_cfg = ExperimentConfig()
    dql_cfg = DistributedQLearningConfig()
    fof_cfg = FriendOrFoeConfig()
    minimax_cfg = MinimaxQConfig()
    fmq_cfg = FMQConfig()
    jal_cfg = JALConfig()

    modes = list(exp_cfg.modes) if args.mode.strip().lower() == "all" else [normalize_mode(args.mode)]
    out_dir = output_dir()
    csv_path = out_dir / "competitive_results.csv"
    env_cfgs = []
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        for mode in modes:
            env_cfgs.append(run_mode(mode, writer, args, exp_cfg.seeds, dql_cfg, fof_cfg, minimax_cfg, fmq_cfg, jal_cfg))

    cfg_path = out_dir / "competitive_run_config.txt"
    cfg_lines = [f"EnvConfig[{i}]=" + str(cfg) for i, cfg in enumerate(env_cfgs)]
    cfg_lines.extend(
        [
            "DistributedQLearningConfig=" + str(asdict(dql_cfg)),
            "FriendOrFoeConfig=" + str(asdict(fof_cfg)),
            "MinimaxQConfig=" + str(asdict(minimax_cfg)),
            "FMQConfig=" + str(asdict(fmq_cfg)),
            "JALConfig=" + str(asdict(jal_cfg)),
            "ExperimentConfig=" + str(asdict(exp_cfg)),
            "modes_run=" + ",".join(modes),
        ]
    )
    cfg_path.write_text("\n".join(cfg_lines), encoding="utf-8")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
