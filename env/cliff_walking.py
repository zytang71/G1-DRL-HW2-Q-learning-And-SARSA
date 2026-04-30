from dataclasses import dataclass


@dataclass(frozen=True)
class StepResult:
    next_state: int
    reward: int
    done: bool
    info: dict


class CliffWalkingEnv:
    ACTION_UP = 0
    ACTION_DOWN = 1
    ACTION_LEFT = 2
    ACTION_RIGHT = 3

    ACTIONS = (ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT)
    ACTION_NAMES = {
        ACTION_UP: "up",
        ACTION_DOWN: "down",
        ACTION_LEFT: "left",
        ACTION_RIGHT: "right",
    }

    def __init__(self, rows: int = 4, cols: int = 12) -> None:
        if rows < 2 or cols < 3:
            raise ValueError("rows must be >= 2 and cols must be >= 3.")

        self.rows = rows
        self.cols = cols
        self.start = (rows - 1, 0)
        self.goal = (rows - 1, cols - 1)
        self.cliff = {(rows - 1, c) for c in range(1, cols - 1)}
        self.state_space = list(range(rows * cols))
        self.action_space = list(self.ACTIONS)
        self.agent_pos = self.start

    @property
    def n_states(self) -> int:
        return self.rows * self.cols

    @property
    def n_actions(self) -> int:
        return len(self.ACTIONS)

    def to_state(self, pos: tuple[int, int]) -> int:
        row, col = pos
        return row * self.cols + col

    def to_pos(self, state: int) -> tuple[int, int]:
        if state < 0 or state >= self.n_states:
            raise ValueError("state index out of range.")
        return divmod(state, self.cols)

    def reset(self) -> int:
        self.agent_pos = self.start
        return self.to_state(self.agent_pos)

    def step(self, action: int) -> StepResult:
        if action not in self.ACTIONS:
            raise ValueError(f"invalid action: {action}")

        row, col = self.agent_pos
        if action == self.ACTION_UP:
            row = max(0, row - 1)
        elif action == self.ACTION_DOWN:
            row = min(self.rows - 1, row + 1)
        elif action == self.ACTION_LEFT:
            col = max(0, col - 1)
        elif action == self.ACTION_RIGHT:
            col = min(self.cols - 1, col + 1)

        candidate = (row, col)
        done = False
        reward = -1
        info: dict = {}

        if candidate in self.cliff:
            reward = -100
            self.agent_pos = self.start
            info["fell_off_cliff"] = True
        else:
            self.agent_pos = candidate
            if candidate == self.goal:
                done = True

        return StepResult(
            next_state=self.to_state(self.agent_pos),
            reward=reward,
            done=done,
            info=info,
        )
