# Phase 7 Analysis

## Learning Performance
- Q-learning convergence episode: 145
- SARSA convergence episode: 382
- Q-learning tail mean reward (last 100): -53.04
- SARSA tail mean reward (last 100): -22.75

## Stability
- Q-learning tail std reward (last 100): 75.94
- SARSA tail std reward (last 100): 19.63
- Lower std indicates more stable learning in late training.

## Final Policy Behavior
- Q-learning greedy path length: 13
- SARSA greedy path length: 17
- Q-learning risk score (distance from cliff): 1.00
- SARSA risk score (distance from cliff): 2.62
- Higher risk score means path stays farther from cliff (more conservative).

## Exploration Effect (epsilon=0.1 evaluation)
- Q-learning cliff falls: 68 / 200 episodes
- SARSA cliff falls: 8 / 200 episodes
- Q-learning avg reward per episode: -52.87
- SARSA avg reward per episode: -22.74

## Generated Artifacts
- `report/total_reward_curve.svg`
- `report/final_policy_paths.svg`
- `report/phase7_metrics.json`
