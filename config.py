from dataclasses import dataclass


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 42
    epsilon: float = 0.1
    alpha: float = 0.1
    gamma: float = 0.9
    episodes: int = 500


DEFAULT_CONFIG = TrainConfig()
