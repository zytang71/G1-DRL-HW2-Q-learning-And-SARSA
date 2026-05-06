import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

from env import CliffWalkingEnv


@dataclass
class AlgorithmData:
    name: str
    rewards: list[float]
    q_table: list[list[float]]


def load_algorithm_data(path: Path) -> AlgorithmData:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return AlgorithmData(
        name=payload["algorithm"],
        rewards=[float(x) for x in payload["episode_rewards"]],
        q_table=[[float(v) for v in row] for row in payload["q_table"]],
    )


def moving_average(values: list[float], window: int) -> list[float]:
    if window <= 0:
        raise ValueError("window must be positive.")
    if not values:
        return []
    ma: list[float] = []
    running = 0.0
    for idx, value in enumerate(values):
        running += value
        if idx >= window:
            running -= values[idx - window]
        current_window = min(idx + 1, window)
        ma.append(running / current_window)
    return ma


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def std(values: list[float]) -> float:
    if not values:
        return 0.0
    m = mean(values)
    return math.sqrt(sum((x - m) ** 2 for x in values) / len(values))


def first_convergence_episode(
    rewards: list[float], ma_window: int = 20, tail_window: int = 50, sustain: int = 20
) -> int:
    if not rewards:
        return -1
    ma = moving_average(rewards, ma_window)
    tail = rewards[-tail_window:] if len(rewards) >= tail_window else rewards[:]
    threshold = mean(tail)
    streak = 0
    for idx, value in enumerate(ma):
        if value >= threshold:
            streak += 1
            if streak >= sustain:
                return idx - sustain + 2  # 1-based episode index
        else:
            streak = 0
    return -1


def argmax_action(q_values: list[float]) -> int:
    best_action = 0
    best_value = q_values[0]
    for action, value in enumerate(q_values[1:], start=1):
        if value > best_value:
            best_value = value
            best_action = action
    return best_action


def greedy_path_from_q(
    env: CliffWalkingEnv, q_table: list[list[float]], max_steps: int = 200
) -> tuple[list[tuple[int, int]], bool]:
    state = env.reset()
    path = [env.to_pos(state)]
    visited: set[int] = set()
    loop_detected = False

    for _ in range(max_steps):
        if state in visited:
            loop_detected = True
            break
        visited.add(state)
        action = argmax_action(q_table[state])
        step = env.step(action)
        state = step.next_state
        path.append(env.to_pos(state))
        if step.done:
            break
    return path, loop_detected


def path_risk_score(env: CliffWalkingEnv, path: list[tuple[int, int]]) -> float:
    cliff_row = env.rows - 1
    distances = []
    for row, _ in path:
        if row == cliff_row:
            continue
        distances.append(cliff_row - row)
    return mean(distances) if distances else 0.0


def epsilon_exploration_stats(
    env: CliffWalkingEnv,
    q_table: list[list[float]],
    epsilon: float,
    episodes: int = 200,
    max_steps: int = 1000,
    seed: int = 42,
) -> dict:
    rng = random.Random(seed)
    total_reward = 0.0
    cliff_falls = 0

    for _ in range(episodes):
        state = env.reset()
        for _ in range(max_steps):
            if rng.random() < epsilon:
                action = rng.randrange(env.n_actions)
            else:
                action = argmax_action(q_table[state])
            step = env.step(action)
            total_reward += step.reward
            if step.info.get("fell_off_cliff"):
                cliff_falls += 1
            state = step.next_state
            if step.done:
                break

    return {
        "episodes": episodes,
        "epsilon": epsilon,
        "avg_reward_per_episode": total_reward / episodes,
        "cliff_falls": cliff_falls,
    }


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def reward_to_polyline_points(
    rewards: list[float], x0: int, y0: int, width: int, height: int, y_min: float, y_max: float
) -> str:
    if not rewards:
        return ""
    points = []
    n = len(rewards)
    span = y_max - y_min if y_max > y_min else 1.0
    for idx, reward in enumerate(rewards):
        x = x0 + (idx / max(n - 1, 1)) * width
        y = y0 + height - ((reward - y_min) / span) * height
        points.append(f"{x:.2f},{y:.2f}")
    return " ".join(points)


def render_reward_curve_svg(
    q_rewards: list[float], s_rewards: list[float], output_path: Path, ma_window: int = 20
) -> None:
    width, height = 1200, 480
    margin_left, margin_top = 70, 30
    plot_w, plot_h = 1080, 380
    y_min = min(min(q_rewards), min(s_rewards))
    y_max = max(max(q_rewards), max(s_rewards))

    q_ma = moving_average(q_rewards, ma_window)
    s_ma = moving_average(s_rewards, ma_window)
    q_points = reward_to_polyline_points(
        q_rewards, margin_left, margin_top, plot_w, plot_h, y_min, y_max
    )
    s_points = reward_to_polyline_points(
        s_rewards, margin_left, margin_top, plot_w, plot_h, y_min, y_max
    )
    q_ma_points = reward_to_polyline_points(
        q_ma, margin_left, margin_top, plot_w, plot_h, y_min, y_max
    )
    s_ma_points = reward_to_polyline_points(
        s_ma, margin_left, margin_top, plot_w, plot_h, y_min, y_max
    )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">
  <rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>
  <text x="{margin_left}" y="20" font-family="Consolas, monospace" font-size="16" fill="#111111">Total Reward per Episode</text>
  <line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_h}" stroke="#333333"/>
  <line x1="{margin_left}" y1="{margin_top + plot_h}" x2="{margin_left + plot_w}" y2="{margin_top + plot_h}" stroke="#333333"/>
  <polyline points="{q_points}" fill="none" stroke="#8ecae6" stroke-width="1" opacity="0.5"/>
  <polyline points="{s_points}" fill="none" stroke="#ffb4a2" stroke-width="1" opacity="0.5"/>
  <polyline points="{q_ma_points}" fill="none" stroke="#0077b6" stroke-width="2.5"/>
  <polyline points="{s_ma_points}" fill="none" stroke="#e76f51" stroke-width="2.5"/>
  <text x="{margin_left + 8}" y="{margin_top + 16}" font-family="Consolas, monospace" font-size="12" fill="#333333">max={y_max:.1f}</text>
  <text x="{margin_left + 8}" y="{margin_top + plot_h - 6}" font-family="Consolas, monospace" font-size="12" fill="#333333">min={y_min:.1f}</text>
  <text x="{margin_left + plot_w - 170}" y="{margin_top + 18}" font-family="Consolas, monospace" font-size="12" fill="#0077b6">Q-learning (MA)</text>
  <text x="{margin_left + plot_w - 170}" y="{margin_top + 36}" font-family="Consolas, monospace" font-size="12" fill="#e76f51">SARSA (MA)</text>
</svg>"""
    write_text(output_path, svg)


def render_path_panel(
    env: CliffWalkingEnv,
    path: list[tuple[int, int]],
    offset_x: int,
    panel_title: str,
) -> str:
    cell = 28
    panel = []
    panel.append(
        f'<text x="{offset_x}" y="24" font-family="Consolas, monospace" font-size="14" fill="#111111">{panel_title}</text>'
    )

    path_set = set(path)
    for row in range(env.rows):
        for col in range(env.cols):
            x = offset_x + col * cell
            y = 40 + row * cell
            pos = (row, col)
            fill = "#f5f5f5"
            if pos == env.start:
                fill = "#90be6d"
            elif pos == env.goal:
                fill = "#577590"
            elif pos in env.cliff:
                fill = "#f94144"
            panel.append(
                f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{fill}" stroke="#cccccc" stroke-width="1"/>'
            )
            if pos in path_set:
                panel.append(
                    f'<circle cx="{x + cell / 2:.1f}" cy="{y + cell / 2:.1f}" r="4" fill="#111111"/>'
                )

    for i in range(len(path) - 1):
        r1, c1 = path[i]
        r2, c2 = path[i + 1]
        x1 = offset_x + c1 * cell + cell / 2
        y1 = 40 + r1 * cell + cell / 2
        x2 = offset_x + c2 * cell + cell / 2
        y2 = 40 + r2 * cell + cell / 2
        panel.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="#111111" stroke-width="1.5"/>'
        )
    return "\n".join(panel)


def render_paths_svg(
    env: CliffWalkingEnv,
    q_path: list[tuple[int, int]],
    s_path: list[tuple[int, int]],
    output_path: Path,
) -> None:
    panel_width = env.cols * 28 + 20
    width = panel_width * 2 + 40
    height = env.rows * 28 + 90

    q_panel = render_path_panel(env, q_path, 20, "Q-learning Greedy Path")
    s_panel = render_path_panel(env, s_path, panel_width + 30, "SARSA Greedy Path")

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">
  <rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>
  {q_panel}
  {s_panel}
</svg>"""
    write_text(output_path, svg)


def main() -> None:
    report_dir = Path("report")
    q_data = load_algorithm_data(report_dir / "q_learning_results.json")
    s_data = load_algorithm_data(report_dir / "sarsa_results.json")
    env = CliffWalkingEnv()

    q_path, q_loop = greedy_path_from_q(env, q_data.q_table)
    s_path, s_loop = greedy_path_from_q(env, s_data.q_table)

    q_tail = q_data.rewards[-100:] if len(q_data.rewards) >= 100 else q_data.rewards[:]
    s_tail = s_data.rewards[-100:] if len(s_data.rewards) >= 100 else s_data.rewards[:]

    q_metrics = {
        "convergence_episode": first_convergence_episode(q_data.rewards),
        "tail_reward_mean": mean(q_tail),
        "tail_reward_std": std(q_tail),
        "greedy_path_length": len(q_path) - 1,
        "greedy_path_loop": q_loop,
        "greedy_path_risk_score": path_risk_score(env, q_path),
    }
    s_metrics = {
        "convergence_episode": first_convergence_episode(s_data.rewards),
        "tail_reward_mean": mean(s_tail),
        "tail_reward_std": std(s_tail),
        "greedy_path_length": len(s_path) - 1,
        "greedy_path_loop": s_loop,
        "greedy_path_risk_score": path_risk_score(env, s_path),
    }

    epsilon = 0.1
    q_explore = epsilon_exploration_stats(env, q_data.q_table, epsilon=epsilon)
    s_explore = epsilon_exploration_stats(env, s_data.q_table, epsilon=epsilon)

    render_reward_curve_svg(
        q_rewards=q_data.rewards,
        s_rewards=s_data.rewards,
        output_path=report_dir / "total_reward_curve.svg",
    )
    render_paths_svg(
        env=env,
        q_path=q_path,
        s_path=s_path,
        output_path=report_dir / "final_policy_paths.svg",
    )

    metrics_payload = {
        "q_learning": q_metrics,
        "sarsa": s_metrics,
        "exploration_eval": {
            "q_learning": q_explore,
            "sarsa": s_explore,
        },
    }
    write_text(
        report_dir / "phase7_metrics.json",
        json.dumps(metrics_payload, indent=2),
    )

    analysis_md = f"""# Phase 7 Analysis

## Learning Performance
- Q-learning convergence episode: {q_metrics["convergence_episode"]}
- SARSA convergence episode: {s_metrics["convergence_episode"]}
- Q-learning tail mean reward (last 100): {q_metrics["tail_reward_mean"]:.2f}
- SARSA tail mean reward (last 100): {s_metrics["tail_reward_mean"]:.2f}

## Stability
- Q-learning tail std reward (last 100): {q_metrics["tail_reward_std"]:.2f}
- SARSA tail std reward (last 100): {s_metrics["tail_reward_std"]:.2f}
- Lower std indicates more stable learning in late training.

## Final Policy Behavior
- Q-learning greedy path length: {q_metrics["greedy_path_length"]}
- SARSA greedy path length: {s_metrics["greedy_path_length"]}
- Q-learning risk score (distance from cliff): {q_metrics["greedy_path_risk_score"]:.2f}
- SARSA risk score (distance from cliff): {s_metrics["greedy_path_risk_score"]:.2f}
- Higher risk score means path stays farther from cliff (more conservative).

## Exploration Effect (epsilon=0.1 evaluation)
- Q-learning cliff falls: {q_explore["cliff_falls"]} / {q_explore["episodes"]} episodes
- SARSA cliff falls: {s_explore["cliff_falls"]} / {s_explore["episodes"]} episodes
- Q-learning avg reward per episode: {q_explore["avg_reward_per_episode"]:.2f}
- SARSA avg reward per episode: {s_explore["avg_reward_per_episode"]:.2f}

## Generated Artifacts
- `report/total_reward_curve.svg`
- `report/final_policy_paths.svg`
- `report/phase7_metrics.json`
"""
    write_text(report_dir / "phase7_analysis.md", analysis_md)

    print("Phase 7 analysis complete.")
    print("Generated: report/total_reward_curve.svg")
    print("Generated: report/final_policy_paths.svg")
    print("Generated: report/phase7_metrics.json")
    print("Generated: report/phase7_analysis.md")


if __name__ == "__main__":
    main()
