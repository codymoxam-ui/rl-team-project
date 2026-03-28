"""
Phase 3.1: Monte Carlo Control (on-policy, first-visit)

Collects a full episode trajectory, then updates Q using first-visit MC
with a constant step-size α (equivalent to incremental mean update):

    G_t  = Σ_{k=0}^{T-t-1} γ^k r_{t+k+1}   (computed backwards)
    Q(s,a) ← Q(s,a) + α [ G_t − Q(s,a) ]    (first visit to (s,a) only)

ε-greedy exploration with exponential decay after each episode.
"""
from typing import List, Tuple

import numpy as np

from .base_agent import BaseAgent


class MonteCarloAgent(BaseAgent):

    def update(
        self,
        state: Tuple[int, int, int],
        action: int,
        reward: float,
        next_state: Tuple[int, int, int],
        done: bool,
    ) -> None:
        # Per-step update is not used in MC; updates happen at episode end.
        pass

    def train(self, n_episodes: int) -> List[float]:
        """
        Run *n_episodes* episodes.  At the end of each episode, apply
        first-visit MC updates backwards through the trajectory.

        Returns per-episode undiscounted returns.
        """
        episode_returns = []

        for _ in range(n_episodes):
            # --- collect trajectory ---
            trajectory = []
            state = self.env.reset()
            done = False
            while not done:
                action = self.select_action(state)
                next_state, reward, done, _ = self.env.step(action)
                trajectory.append((state, action, reward))
                state = next_state

            episode_return = sum(r for _, _, r in trajectory)
            episode_returns.append(episode_return)

            # --- first-visit MC update (backwards pass) ---
            G = 0.0
            visited: set = set()
            for s, a, r in reversed(trajectory):
                G = r + self.gamma * G
                if (s, a) not in visited:
                    visited.add((s, a))
                    site, ps, t = s
                    self.Q[site, ps, t, a] += self.alpha * (G - self.Q[site, ps, t, a])

            self._decay_epsilon()

        return episode_returns
