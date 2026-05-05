from dataclasses import dataclass

from algorithms.common import QTable, TrainingHistory, choose_epsilon_greedy_action
from config import TrainConfig
from env import CliffWalkingEnv


@dataclass
class SARSAResult:
    q_table: QTable
    history: TrainingHistory


def sarsa_target(reward: float, gamma: float, next_q: float, done: bool) -> float:
    if done:
        return reward
    return reward + gamma * next_q


def train_sarsa(
    env: CliffWalkingEnv, config: TrainConfig, max_steps_per_episode: int = 1000
) -> SARSAResult:
    if max_steps_per_episode <= 0:
        raise ValueError("max_steps_per_episode must be positive.")

    q_table = QTable(n_states=env.n_states, n_actions=env.n_actions)
    history = TrainingHistory()

    for _ in range(config.episodes):
        state = env.reset()
        action = choose_epsilon_greedy_action(
            q_values=q_table.values(state), epsilon=config.epsilon
        )
        episode_reward = 0.0
        episode_steps = 0

        for _ in range(max_steps_per_episode):
            step = env.step(action)

            if step.done:
                target = sarsa_target(
                    reward=step.reward, gamma=config.gamma, next_q=0.0, done=True
                )
                q_table.update(
                    state=state, action=action, target=target, alpha=config.alpha
                )
                episode_reward += step.reward
                episode_steps += 1
                break

            next_action = choose_epsilon_greedy_action(
                q_values=q_table.values(step.next_state), epsilon=config.epsilon
            )
            target = sarsa_target(
                reward=step.reward,
                gamma=config.gamma,
                next_q=q_table.get(step.next_state, next_action),
                done=False,
            )
            q_table.update(
                state=state, action=action, target=target, alpha=config.alpha
            )

            episode_reward += step.reward
            episode_steps += 1
            state = step.next_state
            action = next_action

        history.record_episode(total_reward=episode_reward, steps=episode_steps)

    return SARSAResult(q_table=q_table, history=history)
