import random
from dataclasses import dataclass, field


def _validate_epsilon(epsilon: float) -> None:
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("epsilon must be in [0, 1].")


def choose_epsilon_greedy_action(
    q_values: list[float], epsilon: float, rng: random.Random | None = None
) -> int:
    if not q_values:
        raise ValueError("q_values must not be empty.")
    _validate_epsilon(epsilon)

    rand = rng if rng is not None else random
    if rand.random() < epsilon:
        return rand.randrange(len(q_values))

    max_q = max(q_values)
    best_actions = [idx for idx, value in enumerate(q_values) if value == max_q]
    return rand.choice(best_actions)


class QTable:
    def __init__(self, n_states: int, n_actions: int, initial_value: float = 0.0) -> None:
        if n_states <= 0 or n_actions <= 0:
            raise ValueError("n_states and n_actions must be positive.")
        self.n_states = n_states
        self.n_actions = n_actions
        self._table = [
            [float(initial_value) for _ in range(n_actions)] for _ in range(n_states)
        ]

    def _validate_state(self, state: int) -> None:
        if state < 0 or state >= self.n_states:
            raise ValueError("state index out of range.")

    def _validate_action(self, action: int) -> None:
        if action < 0 or action >= self.n_actions:
            raise ValueError("action index out of range.")

    def values(self, state: int) -> list[float]:
        self._validate_state(state)
        return self._table[state][:]

    def get(self, state: int, action: int) -> float:
        self._validate_state(state)
        self._validate_action(action)
        return self._table[state][action]

    def set(self, state: int, action: int, value: float) -> None:
        self._validate_state(state)
        self._validate_action(action)
        self._table[state][action] = float(value)

    def max_value(self, state: int) -> float:
        self._validate_state(state)
        return max(self._table[state])

    def best_action(self, state: int, rng: random.Random | None = None) -> int:
        q_values = self.values(state)
        return choose_epsilon_greedy_action(q_values=q_values, epsilon=0.0, rng=rng)

    def update(self, state: int, action: int, target: float, alpha: float) -> float:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1].")
        old_q = self.get(state, action)
        new_q = old_q + alpha * (target - old_q)
        self.set(state, action, new_q)
        return new_q


@dataclass
class TrainingHistory:
    episode_rewards: list[float] = field(default_factory=list)
    episode_steps: list[int] = field(default_factory=list)

    def record_episode(self, total_reward: float, steps: int) -> None:
        if steps < 0:
            raise ValueError("steps must be >= 0.")
        self.episode_rewards.append(float(total_reward))
        self.episode_steps.append(int(steps))

    @property
    def num_episodes(self) -> int:
        return len(self.episode_rewards)

    def as_dict(self) -> dict:
        return {
            "episode_rewards": self.episode_rewards[:],
            "episode_steps": self.episode_steps[:],
        }
