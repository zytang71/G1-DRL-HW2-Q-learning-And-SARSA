from dataclasses import dataclass

from algorithms.common import QTable, TrainingHistory, choose_epsilon_greedy_action
from config import TrainConfig
from env import CliffWalkingEnv


@dataclass
class QLearningResult:
    q_table: QTable
    history: TrainingHistory


def q_learning_target(reward: float, gamma: float, next_max_q: float, done: bool) -> float:
    if done:
        return reward
    return reward + gamma * next_max_q


def train_q_learning(
    env: CliffWalkingEnv, config: TrainConfig, max_steps_per_episode: int = 1000
) -> QLearningResult:
    if max_steps_per_episode <= 0:
        raise ValueError("max_steps_per_episode must be positive.")

    q_table = QTable(n_states=env.n_states, n_actions=env.n_actions)
    history = TrainingHistory()

    for _ in range(config.episodes):
        state = env.reset()
        episode_reward = 0.0
        episode_steps = 0

        for _ in range(max_steps_per_episode):
            action = choose_epsilon_greedy_action(
                q_values=q_table.values(state), epsilon=config.epsilon
            )
            step = env.step(action)

            target = q_learning_target(
                reward=step.reward,
                gamma=config.gamma,
                next_max_q=q_table.max_value(step.next_state),
                done=step.done,
            )
            q_table.update(
                state=state, action=action, target=target, alpha=config.alpha
            )

            episode_reward += step.reward
            episode_steps += 1
            state = step.next_state

            if step.done:
                break

        history.record_episode(total_reward=episode_reward, steps=episode_steps)

    return QLearningResult(q_table=q_table, history=history)
