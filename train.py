import argparse
import json
import random
from pathlib import Path

from algorithms import train_q_learning, train_sarsa
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
        "--algorithm",
        type=str,
        choices=["q_learning", "sarsa"],
        default="q_learning",
        help="Training algorithm to run.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save training results as JSON.",
    )
    return parser.parse_args()


def default_output_path(algorithm: str) -> str:
    return f"report/{algorithm}_results.json"


def save_training_results(
    output_path: str,
    algorithm: str,
    config: TrainConfig,
    episode_rewards: list[float],
    q_table: list[list[float]],
) -> Path:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "algorithm": algorithm,
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
    if args.algorithm == "q_learning":
        result = train_q_learning(env=env, config=cfg)
    else:
        result = train_sarsa(env=env, config=cfg)

    output_path = args.output if args.output is not None else default_output_path(args.algorithm)
    output_file = save_training_results(
        output_path=output_path,
        algorithm=args.algorithm,
        config=cfg,
        episode_rewards=result.history.episode_rewards,
        q_table=result.q_table.to_nested_list(),
    )

    print(f"Training complete: {args.algorithm}")
    print(
        f"Config(seed={cfg.seed}, epsilon={cfg.epsilon}, alpha={cfg.alpha}, "
        f"gamma={cfg.gamma}, episodes={cfg.episodes})"
    )
    print(f"EpisodesTrained={result.history.num_episodes}")
    print(f"ResultsSaved={output_file.as_posix()}")


if __name__ == "__main__":
    main()
