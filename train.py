import argparse
import random

from config import DEFAULT_CONFIG, TrainConfig

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
    print("Phase 1 initialization complete.")
    print(
        f"Config(seed={cfg.seed}, epsilon={cfg.epsilon}, alpha={cfg.alpha}, "
        f"gamma={cfg.gamma}, episodes={cfg.episodes})"
    )


if __name__ == "__main__":
    main()
