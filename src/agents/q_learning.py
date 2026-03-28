"""
Phase 3.2: Q-Learning (off-policy TD)

Update rule (applied online after every step):
    Q(s,a) ← Q(s,a) + α [ r + γ · max_a' Q(s',a') − Q(s,a) ]

The behaviour policy is ε-greedy; the target policy is greedy.
"""
from typing import Tuple

import numpy as np

from .base_agent import BaseAgent


class QLearningAgent(BaseAgent):

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

        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.Q[ns, nps, nt])

        self.Q[site, ps, t, action] += self.alpha * (target - self.Q[site, ps, t, action])
