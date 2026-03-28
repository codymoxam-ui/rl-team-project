"""
Phase 3.5: Value Iteration (model-based upper-bound baseline)

Uses backward induction on the known finite-horizon MDP:

    V*(s, T)   = 0                          for all terminal states
    Q*(s,a, t) = Σ_{s',r} P(s',r|s,a) [ r + γ V*(s', t+1) ]
    V*(s, t)   = max_a Q*(s,a, t)

Requires env.get_transition_probs() — no interaction with the environment.
Provides an upper-bound benchmark for all model-free agents.
"""
from typing import Optional, Tuple

import numpy as np

from src.env.stimulation_env import N_ACTIONS, N_PATIENT_STATES, N_SITES_STATE


class ValueIterationAgent:
    """
    Parameters
    ----------
    env   : StimulationEnv — must expose get_transition_probs()
    gamma : discount factor (matches model-free agents; default 1.0)
    """

    def __init__(self, env, gamma: float = 1.0) -> None:
        self.env = env
        self.gamma = gamma
        self.horizon = env.horizon
        self._solved = False

        # Value function and Q-table indexed by (site, ps, t [, action])
        self._V = np.zeros((N_SITES_STATE, N_PATIENT_STATES, self.horizon + 1))
        self._Q = np.zeros((N_SITES_STATE, N_PATIENT_STATES, self.horizon + 1, N_ACTIONS))

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------

    def solve(self) -> np.ndarray:
        """
        Run backward induction to compute V* and Q*.

        Returns Q* array of shape (N_SITES_STATE, N_PATIENT_STATES, horizon+1, N_ACTIONS).
        """
        # Terminal values are already 0 (numpy initialisation)
        for t in range(self.horizon - 1, -1, -1):
            for site in range(N_SITES_STATE):
                for ps in range(N_PATIENT_STATES):
                    state = (site, ps, t)
                    for action in range(N_ACTIONS):
                        q_val = 0.0
                        for next_state, reward, prob in self.env.get_transition_probs(state, action):
                            ns, nps, nt = next_state
                            q_val += prob * (reward + self.gamma * self._V[ns, nps, nt])
                        self._Q[site, ps, t, action] = q_val
                    self._V[site, ps, t] = float(np.max(self._Q[site, ps, t]))

        self._solved = True
        return self._Q.copy()

    # ------------------------------------------------------------------
    # Policy
    # ------------------------------------------------------------------

    def get_action(self, state: Tuple[int, int, int]) -> int:
        """Return the greedy action for *state* under the optimal policy."""
        if not self._solved:
            raise RuntimeError("Call solve() before get_action().")
        site, ps, t = state
        return int(np.argmax(self._Q[site, ps, t]))

    def run_episode(self) -> float:
        """Execute one episode with the optimal policy; return total return."""
        if not self._solved:
            raise RuntimeError("Call solve() before run_episode().")
        state = self.env.reset()
        total = 0.0
        done = False
        while not done:
            action = self.get_action(state)
            next_state, reward, done, _ = self.env.step(action)
            total += reward
            state = next_state
        return total

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_Q(self) -> np.ndarray:
        """Return a copy of Q*."""
        return self._Q.copy()

    def get_V(self) -> np.ndarray:
        """Return a copy of V*."""
        return self._V.copy()
