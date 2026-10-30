"""Additional MARL tabular algorithms for Section 5-1.

Implemented algorithms:
- Distributed Q-Learning (DQL)
- Friend-or-Foe Q-Learning (FoF-Q)
- Minimax Q-Learning
- Frequency Maximum Q-Learning (FMQ)
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import DefaultDict, Dict, List, Tuple
import numpy as np

from environment_discrete_multi import MultiAgentDiscreteSoccerEnv
from independent_q_learning import EpisodeLog, linear_epsilon


State = Tuple[int, ...]


class TeamHeuristic:
    def action(self, env: MultiAgentDiscreteSoccerEnv, aid: str) -> int:
        pos = env.positions[aid]
        ball = env.ball_pos
        if env.ball_owner == aid:
            return env.ACTION_KICK

        dx = int(ball[0] - pos[0])
        dy = int(ball[1] - pos[1])
        if abs(dx) > abs(dy):
            return env.ACTION_RIGHT if dx > 0 else env.ACTION_LEFT
        if dy != 0:
            return env.ACTION_DOWN if dy > 0 else env.ACTION_UP
        return env.ACTION_STAY


def _mean_team_reward(env: MultiAgentDiscreteSoccerEnv, rewards: Dict[str, float], team_id: int) -> float:
    return float(np.mean([rewards[aid] for aid in env.team_agents(team_id)]))


def _base_episode_log(
    algorithm: str,
    setting: str,
    mode: str,
    seed: int,
    ep: int,
    team_reward: Dict[int, float],
    info_last: Dict[str, float],
    steps: int,
    td_vals: List[float],
) -> EpisodeLog:
    return EpisodeLog(
        algorithm=algorithm,
        setting=setting,
        mode=mode,
        seed=seed,
        episode=ep,
        team0_reward=float(team_reward[0]),
        team1_reward=float(team_reward[1]),
        team0_scored=int(info_last.get("team0_scored", 0.0) > 0.5),
        team1_scored=int(info_last.get("team1_scored", 0.0) > 0.5),
        episode_length=steps,
        mean_abs_td_error=float(np.mean(td_vals) if td_vals else 0.0),
    )


class DistributedQLearner:
    """Independent team-0 learners with consensus averaging after each update."""

    def __init__(self, env: MultiAgentDiscreteSoccerEnv, alpha: float, gamma: float, consensus_tau: float) -> None:
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.consensus_tau = float(np.clip(consensus_tau, 0.0, 1.0))
        self.team0_agents = env.team_agents(0)
        self.team1_agents = env.team_agents(1)
        self.q_tables: Dict[str, DefaultDict[State, np.ndarray]] = {
            aid: defaultdict(lambda: np.zeros(env.action_space_n, dtype=np.float64)) for aid in self.team0_agents
        }
        self.heur = TeamHeuristic()

    def choose_actions(self, obs: Dict[str, State], epsilon: float, rng: np.random.Generator) -> Dict[str, int]:
        acts: Dict[str, int] = {}
        for aid in self.team0_agents:
            q = self.q_tables[aid][obs[aid]]
            if rng.random() < epsilon:
                acts[aid] = int(rng.integers(0, self.env.action_space_n))
            else:
                acts[aid] = int(np.argmax(q))
                
        for aid in self.team1_agents:
            acts[aid] = self.heur.action(self.env, aid)
        return acts

    def update(
        self,
        obs: Dict[str, State],
        acts: Dict[str, int],
        rewards: Dict[str, float],
        next_obs: Dict[str, State],
        done: bool,
    ) -> float:
        td_vals = []
        for aid in self.team0_agents:
            s = obs[aid]
            a = acts[aid]
            r = rewards[aid]
            ns = next_obs[aid]
            q_sa = self.q_tables[aid][s][a]
            target = r + self.gamma * (0.0 if done else np.max(self.q_tables[aid][ns]))
            td = target - q_sa
            self.q_tables[aid][s][a] += self.alpha * td
            td_vals.append(abs(td))

        if len(self.team0_agents) > 1 and self.consensus_tau > 0.0:
            states = [obs[aid] for aid in self.team0_agents]
            unique_states = list(dict.fromkeys(states))
            for s in unique_states:
                q_stack = np.stack([self.q_tables[aid][s] for aid in self.team0_agents], axis=0)
                q_mean = np.mean(q_stack, axis=0)
                for aid in self.team0_agents:
                    self.q_tables[aid][s] = (1.0 - self.consensus_tau) * self.q_tables[aid][s] + self.consensus_tau * q_mean

        return float(np.mean(td_vals) if td_vals else 0.0)


def train_distributed_q_learning(
    env: MultiAgentDiscreteSoccerEnv,
    episodes: int,
    alpha: float,
    gamma: float,
    epsilon_start: float,
    epsilon_end: float,
    epsilon_decay_episodes: int,
    consensus_tau: float,
    seed: int,
    setting: str,
    mode: str,
) -> List[EpisodeLog]:
    rng = np.random.default_rng(seed)
    learner = DistributedQLearner(env=env, alpha=alpha, gamma=gamma, consensus_tau=consensus_tau)
    logs: List[EpisodeLog] = []

    for ep in range(episodes):
        eps = linear_epsilon(ep, epsilon_start, epsilon_end, epsilon_decay_episodes)
        obs = env.reset(seed=seed * 100_000 + ep)
        done = False
        team_reward = {0: 0.0, 1: 0.0}
        steps = 0
        td_vals: List[float] = []
        info_last = {"team0_scored": 0.0, "team1_scored": 0.0}

        while not done and steps < env.max_steps:
            acts = learner.choose_actions(obs, eps, rng)
            out = env.step(acts)
            td = learner.update(obs, acts, out.rewards, out.observations, out.done)

            for aid, rew in out.rewards.items():
                team_reward[env.team_of(aid)] += rew

            td_vals.append(td)
            obs = out.observations
            done = out.done
            info_last = out.info
            steps += 1

        logs.append(_base_episode_log("distributed_q_learning", setting, mode, seed, ep, team_reward, info_last, steps, td_vals))

    return logs


class TwoPlayerMatrixLearner:
    """Shared mechanics for FoF-Q and Minimax-Q with one learner vs one opponent."""

    def __init__(self, env: MultiAgentDiscreteSoccerEnv, alpha: float, gamma: float) -> None:
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.team0_agents = env.team_agents(0)
        self.team1_agents = env.team_agents(1)
        self.learner_aid = self.team0_agents[0]
        self.opponent_aid = self.team1_agents[0]
        self.n_actions = env.action_space_n
        self.q: DefaultDict[State, np.ndarray] = defaultdict(
            lambda: np.zeros((self.n_actions, self.n_actions), dtype=np.float64)
        )
        self.heur = TeamHeuristic()

    def state(self, obs: Dict[str, State]) -> State:
        return tuple(obs[self.learner_aid] + obs[self.opponent_aid])

    def build_actions(self, action_learner: int) -> Tuple[Dict[str, int], int]:
        acts: Dict[str, int] = {}
        acts[self.learner_aid] = int(action_learner)
        for aid in self.team0_agents[1:]:
            acts[aid] = self.heur.action(self.env, aid)
        for aid in self.team1_agents:
            acts[aid] = self.heur.action(self.env, aid)
        return acts, int(acts[self.opponent_aid])

    def minimax_value(self, q_mat: np.ndarray) -> float:
        return float(np.max(np.min(q_mat, axis=1)))

    def minimax_action(self, q_mat: np.ndarray) -> int:
        return int(np.argmax(np.min(q_mat, axis=1)))

    def friend_action(self, q_mat: np.ndarray) -> int:
        return int(np.argmax(np.max(q_mat, axis=1)))


def train_friend_or_foe_q(
    env: MultiAgentDiscreteSoccerEnv,
    episodes: int,
    alpha: float,
    gamma: float,
    epsilon_start: float,
    epsilon_end: float,
    epsilon_decay_episodes: int,
    seed: int,
    setting: str,
    mode: str,
) -> List[EpisodeLog]:
    rng = np.random.default_rng(seed)
    learner = TwoPlayerMatrixLearner(env=env, alpha=alpha, gamma=gamma)
    logs: List[EpisodeLog] = []
    cooperative = setting.lower().startswith("coop")

    for ep in range(episodes):
        eps = linear_epsilon(ep, epsilon_start, epsilon_end, epsilon_decay_episodes)
        obs = env.reset(seed=seed * 100_000 + ep)
        done = False
        team_reward = {0: 0.0, 1: 0.0}
        steps = 0
        td_vals: List[float] = []
        info_last = {"team0_scored": 0.0, "team1_scored": 0.0}

        while not done and steps < env.max_steps:
            s = learner.state(obs)
            q_mat = learner.q[s]
            greedy_a = learner.friend_action(q_mat) if cooperative else learner.minimax_action(q_mat)
            a = int(rng.integers(0, learner.n_actions)) if rng.random() < eps else greedy_a

            acts, b = learner.build_actions(a)
            out = env.step(acts)
            ns = learner.state(out.observations)
            r = _mean_team_reward(env, out.rewards, team_id=0)

            next_q = learner.q[ns]
            next_v = float(np.max(next_q)) if cooperative else learner.minimax_value(next_q)
            q_sa = learner.q[s][a, b]
            target = r + gamma * (0.0 if out.done else next_v)
            td = target - q_sa
            learner.q[s][a, b] += alpha * td

            for aid, rew in out.rewards.items():
                team_reward[env.team_of(aid)] += rew

            td_vals.append(abs(td))
            obs = out.observations
            done = out.done
            info_last = out.info
            steps += 1

        logs.append(_base_episode_log("friend_or_foe_q", setting, mode, seed, ep, team_reward, info_last, steps, td_vals))

    return logs


def train_minimax_q_learning(
    env: MultiAgentDiscreteSoccerEnv,
    episodes: int,
    alpha: float,
    gamma: float,
    epsilon_start: float,
    epsilon_end: float,
    epsilon_decay_episodes: int,
    seed: int,
    setting: str,
    mode: str,
) -> List[EpisodeLog]:
    rng = np.random.default_rng(seed)
    learner = TwoPlayerMatrixLearner(env=env, alpha=alpha, gamma=gamma)
    logs: List[EpisodeLog] = []

    for ep in range(episodes):
        eps = linear_epsilon(ep, epsilon_start, epsilon_end, epsilon_decay_episodes)
        obs = env.reset(seed=seed * 100_000 + ep)
        done = False
        team_reward = {0: 0.0, 1: 0.0}
        steps = 0
        td_vals: List[float] = []
        info_last = {"team0_scored": 0.0, "team1_scored": 0.0}

        while not done and steps < env.max_steps:
            s = learner.state(obs)
            q_mat = learner.q[s]
            greedy_a = learner.minimax_action(q_mat)
            a = int(rng.integers(0, learner.n_actions)) if rng.random() < eps else greedy_a

            acts, b = learner.build_actions(a)
            out = env.step(acts)
            ns = learner.state(out.observations)
            r = _mean_team_reward(env, out.rewards, team_id=0)

            q_sa = learner.q[s][a, b]
            next_v = learner.minimax_value(learner.q[ns])
            target = r + gamma * (0.0 if out.done else next_v)
            td = target - q_sa
            learner.q[s][a, b] += alpha * td

            for aid, rew in out.rewards.items():
                team_reward[env.team_of(aid)] += rew

            td_vals.append(abs(td))
            obs = out.observations
            done = out.done
            info_last = out.info
            steps += 1

        logs.append(_base_episode_log("minimax_q_learning", setting, mode, seed, ep, team_reward, info_last, steps, td_vals))

    return logs


class FMQLearner:
    def __init__(self, env: MultiAgentDiscreteSoccerEnv, alpha: float, gamma: float, fmq_c: float) -> None:
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.fmq_c = fmq_c
        self.team0_agents = env.team_agents(0)
        self.team1_agents = env.team_agents(1)
        self.learner_aid = self.team0_agents[0]
        self.q: DefaultDict[State, np.ndarray] = defaultdict(lambda: np.zeros(env.action_space_n, dtype=np.float64))
        self.n_sa: DefaultDict[State, np.ndarray] = defaultdict(lambda: np.zeros(env.action_space_n, dtype=np.float64))
        self.n_max: DefaultDict[State, np.ndarray] = defaultdict(lambda: np.zeros(env.action_space_n, dtype=np.float64))
        self.max_reward_seen: DefaultDict[State, float] = defaultdict(lambda: -np.inf)
        self.heur = TeamHeuristic()

    def score(self, state: State) -> np.ndarray:
        q = self.q[state]
        freq = self.n_max[state] / np.maximum(self.n_sa[state], 1.0)
        return q + self.fmq_c * freq * np.max(q)

    def choose_action(self, state: State, epsilon: float, rng: np.random.Generator) -> int:
        if rng.random() < epsilon:
            return int(rng.integers(0, self.env.action_space_n))
        return int(np.argmax(self.score(state)))

    def build_actions(self, action_learner: int) -> Dict[str, int]:
        acts: Dict[str, int] = {self.learner_aid: int(action_learner)}
        for aid in self.team0_agents[1:]:
            acts[aid] = self.heur.action(self.env, aid)
        for aid in self.team1_agents:
            acts[aid] = self.heur.action(self.env, aid)
        return acts

    def update(self, s: State, a: int, r: float, ns: State, done: bool) -> float:
        q_sa = self.q[s][a]
        target = r + self.gamma * (0.0 if done else np.max(self.q[ns]))
        td = target - q_sa
        self.q[s][a] += self.alpha * td

        self.n_sa[s][a] += 1.0
        if r > self.max_reward_seen[s]:
            self.max_reward_seen[s] = r
            self.n_max[s][:] = 0.0
            self.n_max[s][a] = 1.0
        elif np.isclose(r, self.max_reward_seen[s]):
            self.n_max[s][a] += 1.0

        return float(abs(td))


def train_fmq(
    env: MultiAgentDiscreteSoccerEnv,
    episodes: int,
    alpha: float,
    gamma: float,
    epsilon_start: float,
    epsilon_end: float,
    epsilon_decay_episodes: int,
    fmq_c: float,
    seed: int,
    setting: str,
    mode: str,
) -> List[EpisodeLog]:
    rng = np.random.default_rng(seed)
    learner = FMQLearner(env=env, alpha=alpha, gamma=gamma, fmq_c=fmq_c)
    logs: List[EpisodeLog] = []

    for ep in range(episodes):
        eps = linear_epsilon(ep, epsilon_start, epsilon_end, epsilon_decay_episodes)
        obs = env.reset(seed=seed * 100_000 + ep)
        done = False
        team_reward = {0: 0.0, 1: 0.0}
        steps = 0
        td_vals: List[float] = []
        info_last = {"team0_scored": 0.0, "team1_scored": 0.0}

        while not done and steps < env.max_steps:
            s = obs[learner.learner_aid]
            a = learner.choose_action(s, eps, rng)
            acts = learner.build_actions(a)
            out = env.step(acts)
            ns = out.observations[learner.learner_aid]
            r = _mean_team_reward(env, out.rewards, team_id=0)
            td = learner.update(s, a, r, ns, out.done)

            for aid, rew in out.rewards.items():
                team_reward[env.team_of(aid)] += rew

            td_vals.append(td)
            obs = out.observations
            done = out.done
            info_last = out.info
            steps += 1

        logs.append(_base_episode_log("fmq", setting, mode, seed, ep, team_reward, info_last, steps, td_vals))

    return logs
