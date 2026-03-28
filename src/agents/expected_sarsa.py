"""
Phase 3.3: Expected SARSA (on-policy TD)

Update rule (applied online after every step):
    Q(s,a) ← Q(s,a) + α [ r + γ · Σ_a' π(a'|s') Q(s',a') − Q(s,a) ]

Expected value under ε-greedy π:
    V(s') = (ε / |A|) · Σ_a Q(s',a)  +  (1−ε) · max_a Q(s',a)
"""
from typing import Tuple

import numpy as np

from src.env.stimulation_env import N_ACTIONS
from .base_agent import BaseAgent


class ExpectedSARSAAgent(BaseAgent):

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
            q_next = self.Q[ns, nps, nt]
            # Expected value under ε-greedy policy
            best_val = float(np.max(q_next))
            expected = (self.epsilon / N_ACTIONS) * float(np.sum(q_next)) \
                       + (1.0 - self.epsilon) * best_val
            target = reward + self.gamma * expected

        self.Q[site, ps, t, action] += self.alpha * (target - self.Q[site, ps, t, action])
