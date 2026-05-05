import argparse
import json
import random
from pathlib import Path

from algorithms import train_q_learning
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
        description="Training entrypoint for Q-learning and SARSA."
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Override training episodes (must be >= 500 for assignment).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="report/q_learning_results.json",
        help="Path to save Q-learning results as JSON.",
    )
    return parser.parse_args()


def save_q_learning_results(
    output_path: str, config: TrainConfig, episode_rewards: list[float], q_table: list[list[float]]
) -> Path:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "algorithm": "q_learning",
        "config": {
            "seed": config.seed,
            "epsilon": config.epsilon,
            "alpha": config.alpha,
            "gamma": config.gamma,
            "episodes": config.episodes,
        },
        "episode_rewards": episode_rewards,
        "q_table": q_table,
    }
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return destination


def main() -> None:
    args = parse_args()
    cfg = load_config(args.episodes)

    if cfg.episodes < 500:
        raise ValueError("episodes must be >= 500 for this assignment.")

    set_global_seed(cfg.seed)
    env = CliffWalkingEnv()
    result = train_q_learning(env=env, config=cfg)
    output_file = save_q_learning_results(
        output_path=args.output,
        config=cfg,
        episode_rewards=result.history.episode_rewards,
        q_table=result.q_table.to_nested_list(),
    )

    print("Phase 4 Q-learning training complete.")
    print(
        f"Config(seed={cfg.seed}, epsilon={cfg.epsilon}, alpha={cfg.alpha}, "
        f"gamma={cfg.gamma}, episodes={cfg.episodes})"
    )
    print(f"EpisodesTrained={result.history.num_episodes}")
    print(f"ResultsSaved={output_file.as_posix()}")


if __name__ == "__main__":
    main()
