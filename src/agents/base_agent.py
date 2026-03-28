"""
Phase 3.0: Base Agent

Abstract base class shared by all model-free RL agents.
Provides the Q-table, ε-greedy action selection, ε decay, and the
default TD-style training loop.  Monte Carlo overrides train().
"""
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import numpy as np

from src.env.stimulation_env import N_ACTIONS, N_PATIENT_STATES, N_SITES_STATE


class BaseAgent(ABC):
    """
    Parameters
    ----------
    env           : StimulationEnv instance
    alpha         : constant step-size (learning rate)
    epsilon       : initial exploration probability
    epsilon_decay : multiplicative decay applied after each episode
    epsilon_min   : floor on epsilon
    gamma         : discount factor (1.0 for finite-horizon total return)
    seed          : RNG seed for reproducibility
    """

    def __init__(
        self,
        env,
        alpha: float = 0.1,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.05,
        gamma: float = 1.0,
        seed: Optional[int] = None,
    ) -> None:
        if not 0 < alpha <= 1:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        if not 0 <= epsilon <= 1:
            raise ValueError(f"epsilon must be in [0, 1], got {epsilon}")
        if not 0 < epsilon_decay <= 1:
            raise ValueError(f"epsilon_decay must be in (0, 1], got {epsilon_decay}")
        if not 0 <= epsilon_min <= epsilon:
            raise ValueError(f"epsilon_min must be in [0, epsilon], got {epsilon_min}")
        if not 0 < gamma <= 1:
            raise ValueError(f"gamma must be in (0, 1], got {gamma}")

        self.env = env
        self.alpha = alpha
        self.epsilon = epsilon
        self._epsilon_0 = epsilon        # saved for reset()
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.gamma = gamma
        self._rng = np.random.default_rng(seed)

        # Q-table: shape (site, patient_state, t, action)
        self.Q = np.zeros(
            (N_SITES_STATE, N_PATIENT_STATES, env.horizon + 1, N_ACTIONS)
        )

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(self, state: Tuple[int, int, int]) -> int:
        """ε-greedy action selection."""
        if self._rng.random() < self.epsilon:
            return int(self._rng.integers(N_ACTIONS))
        site, ps, t = state
        return int(np.argmax(self.Q[site, ps, t]))

    # ------------------------------------------------------------------
    # Update (must be implemented by each subclass)
    # ------------------------------------------------------------------

    @abstractmethod
    def update(
        self,
        state: Tuple[int, int, int],
        action: int,
        reward: float,
        next_state: Tuple[int, int, int],
        done: bool,
    ) -> None:
        """Update Q-table given one (s, a, r, s', done) transition."""

    # ------------------------------------------------------------------
    # Training loop (TD-style; Monte Carlo overrides this)
    # ------------------------------------------------------------------

    def train(self, n_episodes: int) -> List[float]:
        """
        Run *n_episodes* episodes with the current policy.

        Returns a list of per-episode undiscounted returns.
        """
        episode_returns = []
        for _ in range(n_episodes):
            state = self.env.reset()
            episode_return = 0.0
            done = False
            while not done:
                action = self.select_action(state)
                next_state, reward, done, _ = self.env.step(action)
                self.update(state, action, reward, next_state, done)
                state = next_state
                episode_return += reward
            self._decay_epsilon()
            episode_returns.append(episode_return)
        return episode_returns

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _decay_epsilon(self) -> None:
        """Apply one step of exponential ε decay."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_Q(self) -> np.ndarray:
        """Return a copy of the Q-table."""
        return self.Q.copy()

    def reset(self) -> None:
        """Reset Q-table and epsilon to initial values (for a new seed run)."""
        self.Q[:] = 0.0
        self.epsilon = self._epsilon_0
