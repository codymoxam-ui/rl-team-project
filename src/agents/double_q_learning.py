"""
Phase 3.4: Double Q-Learning (off-policy, reduced maximisation bias)

Maintains two independent Q-tables Q1 and Q2.
At each step, with probability 0.5:
  - Update Q1: target = r + γ · Q2(s', argmax_a Q1(s',a))
  - Update Q2: target = r + γ · Q1(s', argmax_a Q2(s',a))

Action selection uses the average Q1 + Q2.
"""
from typing import Tuple

import numpy as np

from src.env.stimulation_env import N_ACTIONS, N_PATIENT_STATES, N_SITES_STATE
from .base_agent import BaseAgent


class DoubleQLearningAgent(BaseAgent):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # self.Q from BaseAgent serves as Q1
        self.Q1 = self.Q
        self.Q2 = np.zeros_like(self.Q1)

    def select_action(self, state: Tuple[int, int, int]) -> int:
        """ε-greedy on the average of Q1 and Q2."""
        if self._rng.random() < self.epsilon:
            return int(self._rng.integers(N_ACTIONS))
        site, ps, t = state
        return int(np.argmax(self.Q1[site, ps, t] + self.Q2[site, ps, t]))

    def update(
        self,
        state: Tuple[int, int, int],
        action: int,
        reward: float,
        next_state: Tuple[int, int, int],
        done: bool,
    ) -> None:
        site, ps, t = state
        ns, nps, nt = next_state

        if self._rng.random() < 0.5:
            # Update Q1, bootstrap from Q2
            if done:
                target = reward
            else:
                best_a = int(np.argmax(self.Q1[ns, nps, nt]))
                target = reward + self.gamma * self.Q2[ns, nps, nt, best_a]
            self.Q1[site, ps, t, action] += self.alpha * (target - self.Q1[site, ps, t, action])
        else:
            # Update Q2, bootstrap from Q1
            if done:
                target = reward
            else:
                best_a = int(np.argmax(self.Q2[ns, nps, nt]))
                target = reward + self.gamma * self.Q1[ns, nps, nt, best_a]
            self.Q2[site, ps, t, action] += self.alpha * (target - self.Q2[site, ps, t, action])

    def get_Q(self) -> np.ndarray:
        """Return average of Q1 and Q2."""
        return ((self.Q1 + self.Q2) / 2.0).copy()

    def reset(self) -> None:
        """Reset both Q-tables and epsilon."""
        super().reset()   # zeroes Q (= Q1) and resets epsilon
        self.Q2[:] = 0.0
