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
        choices=["q_learning", "sarsa", "both"],
        default="q_learning",
        help="Training algorithm to run. Use 'both' for fair comparison.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save training results as JSON.",
    )
    parser.add_argument(
        "--max-steps-per-episode",
        type=int,
        default=1000,
        help="Shared step limit per episode for all algorithms.",
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


def save_fair_comparison_summary(
    output_path: str,
    config: TrainConfig,
    max_steps_per_episode: int,
    env_shape: tuple[int, int],
    q_learning_output: Path,
    sarsa_output: Path,
) -> Path:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows, cols = env_shape
    payload = {
        "mode": "fair_comparison",
        "shared_config": {
            "seed": config.seed,
            "epsilon": config.epsilon,
            "alpha": config.alpha,
            "gamma": config.gamma,
            "episodes": config.episodes,
            "max_steps_per_episode": max_steps_per_episode,
            "env_rows": rows,
            "env_cols": cols,
        },
        "outputs": {
            "q_learning": q_learning_output.as_posix(),
            "sarsa": sarsa_output.as_posix(),
        },
    }
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return destination


def main() -> None:
    args = parse_args()
    cfg = load_config(args.episodes)

    if cfg.episodes < 500:
        raise ValueError("episodes must be >= 500 for this assignment.")
    if args.max_steps_per_episode <= 0:
        raise ValueError("max_steps_per_episode must be positive.")

    env_template = CliffWalkingEnv()
    env_shape = (env_template.rows, env_template.cols)

    if args.algorithm == "both":
        set_global_seed(cfg.seed)
        q_env = CliffWalkingEnv()
        q_result = train_q_learning(
            env=q_env, config=cfg, max_steps_per_episode=args.max_steps_per_episode
        )
        q_output = save_training_results(
            output_path=default_output_path("q_learning"),
            algorithm="q_learning",
            config=cfg,
            episode_rewards=q_result.history.episode_rewards,
            q_table=q_result.q_table.to_nested_list(),
        )

        set_global_seed(cfg.seed)
        s_env = CliffWalkingEnv()
        s_result = train_sarsa(
            env=s_env, config=cfg, max_steps_per_episode=args.max_steps_per_episode
        )
        s_output = save_training_results(
            output_path=default_output_path("sarsa"),
            algorithm="sarsa",
            config=cfg,
            episode_rewards=s_result.history.episode_rewards,
            q_table=s_result.q_table.to_nested_list(),
        )

        summary_output = args.output or "report/fair_comparison_summary.json"
        summary_file = save_fair_comparison_summary(
            output_path=summary_output,
            config=cfg,
            max_steps_per_episode=args.max_steps_per_episode,
            env_shape=env_shape,
            q_learning_output=q_output,
            sarsa_output=s_output,
        )
        print("Training complete: fair comparison (q_learning + sarsa)")
        print(
            f"Config(seed={cfg.seed}, epsilon={cfg.epsilon}, alpha={cfg.alpha}, "
            f"gamma={cfg.gamma}, episodes={cfg.episodes})"
        )
        print(f"QlearningEpisodes={q_result.history.num_episodes}")
        print(f"SARSAEpisodes={s_result.history.num_episodes}")
        print(f"SummarySaved={summary_file.as_posix()}")
        return

    set_global_seed(cfg.seed)
    env = CliffWalkingEnv()
    if args.algorithm == "q_learning":
        result = train_q_learning(
            env=env, config=cfg, max_steps_per_episode=args.max_steps_per_episode
        )
    else:
        result = train_sarsa(
            env=env, config=cfg, max_steps_per_episode=args.max_steps_per_episode
        )

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
