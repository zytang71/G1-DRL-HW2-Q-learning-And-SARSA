import argparse
import random

from algorithms import QTable, TrainingHistory, choose_epsilon_greedy_action
from config import DEFAULT_CONFIG, TrainConfig
from env import CliffWalkingEnv

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)


def load_config(episodes: int | None) -> TrainConfig:
    if episodes is None:
        return DEFAULT_CONFIG
    return TrainConfig(
        seed=DEFAULT_CONFIG.seed,
        epsilon=DEFAULT_CONFIG.epsilon,
        alpha=DEFAULT_CONFIG.alpha,
        gamma=DEFAULT_CONFIG.gamma,
        episodes=episodes,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Training entrypoint for Q-learning and SARSA (phase 1 scaffold)."
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Override training episodes (must be >= 500 for assignment).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.episodes)

    if cfg.episodes < 500:
        raise ValueError("episodes must be >= 500 for this assignment.")

    set_global_seed(cfg.seed)
    env = CliffWalkingEnv()
    q_table = QTable(n_states=env.n_states, n_actions=env.n_actions)
    history = TrainingHistory()

    state = env.reset()
    action = choose_epsilon_greedy_action(q_table.values(state), epsilon=cfg.epsilon)
    step = env.step(action)
    q_table.update(state=state, action=action, target=step.reward, alpha=cfg.alpha)
    history.record_episode(total_reward=step.reward, steps=1)

    print("Phase 1-3 scaffolding complete.")
    print(
        f"Config(seed={cfg.seed}, epsilon={cfg.epsilon}, alpha={cfg.alpha}, "
        f"gamma={cfg.gamma}, episodes={cfg.episodes})"
    )
    print(
        f"SmokeTest(state={state}, action={action}, reward={step.reward}, "
        f"done={step.done}, history_episodes={history.num_episodes})"
    )


if __name__ == "__main__":
    main()
