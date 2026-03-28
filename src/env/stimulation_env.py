"""
Phase 2: Stimulation MDP Environment

Finite-horizon MDP for adaptive multi-site brain stimulation.
The agent chooses which of 4 sites (S1-S4) to stimulate at each timestep
to maximise cumulative EEG reward while managing stochastic patient state.
"""
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

SITES = ['S1', 'S2', 'S3', 'S4']                        # action indices 0-3
PATIENT_STATES = ['baseline', 'receptive', 'non_receptive']  # indices 0-2
OBSERVATIONS = ['favorable', 'neutral', 'unfavorable']       # indices 0-2
REWARD_MAP = {'favorable': +1, 'neutral': 0, 'unfavorable': -1}

N_SITES_STATE = 5    # state site dim: 0=Start, 1=S1, 2=S2, 3=S3, 4=S4
N_PATIENT_STATES = 3
N_ACTIONS = 4
N_OBS = 3

VALID_SETTINGS = ('high', 'moderate', 'low')

# ---------------------------------------------------------------------------
# Patient state transition model
# T[(patient_state_idx, is_same_site)] -> np.array([p_baseline, p_receptive, p_non_receptive])
# Baseline transitions identically for same/different site.
# ---------------------------------------------------------------------------
PATIENT_TRANSITION: Dict[Tuple[int, bool], np.ndarray] = {
    (0, False): np.array([0.60, 0.30, 0.10]),
    (0, True):  np.array([0.60, 0.30, 0.10]),
    (1, True):  np.array([0.20, 0.30, 0.50]),
    (1, False): np.array([0.50, 0.30, 0.20]),
    (2, True):  np.array([0.10, 0.10, 0.80]),
    (2, False): np.array([0.40, 0.30, 0.30]),
}

# ---------------------------------------------------------------------------
# Observation model
# OBS_PROBS[setting][site_action_idx, patient_state_idx] -> [p_fav, p_neu, p_unf]
# shape: (4, 3, 3)  axes: (site_action, patient_state, obs)
# ---------------------------------------------------------------------------
OBS_PROBS: Dict[str, np.ndarray] = {
    'high': np.array([
        # S1: highest-value site, degrades sharply when non-receptive
        [[0.70, 0.20, 0.10], [0.85, 0.10, 0.05], [0.30, 0.40, 0.30]],
        # S2: intermediate site
        [[0.50, 0.30, 0.20], [0.65, 0.25, 0.10], [0.20, 0.40, 0.40]],
        # S3: weakest site, useful for exploration only
        [[0.30, 0.40, 0.30], [0.45, 0.35, 0.20], [0.15, 0.35, 0.50]],
        # S4: secondary high-value site, more robust under repeated use
        [[0.45, 0.35, 0.20], [0.60, 0.30, 0.10], [0.25, 0.40, 0.35]],
    ]),
    'moderate': np.array([
        # Sites differ moderately; state effects remain important
        [[0.60, 0.25, 0.15], [0.75, 0.15, 0.10], [0.25, 0.40, 0.35]],
        [[0.50, 0.30, 0.20], [0.60, 0.25, 0.15], [0.20, 0.40, 0.40]],
        [[0.40, 0.35, 0.25], [0.50, 0.30, 0.20], [0.20, 0.40, 0.40]],
        [[0.45, 0.35, 0.20], [0.55, 0.30, 0.15], [0.20, 0.40, 0.40]],
    ]),
    'low': np.array([
        # Sites nearly identical; patient-state dynamics drive policy
        [[0.45, 0.35, 0.20], [0.60, 0.25, 0.15], [0.20, 0.40, 0.40]],
        [[0.40, 0.35, 0.25], [0.55, 0.30, 0.15], [0.20, 0.40, 0.40]],
        [[0.40, 0.35, 0.25], [0.55, 0.30, 0.15], [0.20, 0.40, 0.40]],
        [[0.40, 0.35, 0.25], [0.55, 0.30, 0.15], [0.20, 0.40, 0.40]],
    ]),
}


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class StimulationEnv:
    """
    Finite-horizon MDP for adaptive multi-site brain stimulation.

    State tuple: (current_site_idx, patient_state_idx, t)
        current_site_idx : 0=Start, 1=S1, 2=S2, 3=S3, 4=S4
        patient_state_idx: 0=baseline, 1=receptive, 2=non_receptive
        t                : 0 .. horizon  (t==horizon is terminal)

    Actions: integers 0-3  (0=S1, 1=S2, 2=S3, 3=S4)

    Parameters
    ----------
    setting   : 'high' | 'moderate' | 'low'  — EEG response separation
    horizon   : episode length I (number of stimulation decisions)
    c_switch  : penalty applied when switching away from the current site
    seed      : RNG seed for reproducibility
    """

    def __init__(
        self,
        setting: str = 'high',
        horizon: int = 10,
        c_switch: float = 0.0,
        seed: Optional[int] = None,
    ) -> None:
        if setting not in VALID_SETTINGS:
            raise ValueError(f"setting must be one of {VALID_SETTINGS}, got '{setting}'")
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}")
        if c_switch < 0:
            raise ValueError(f"c_switch must be >= 0, got {c_switch}")

        self.setting = setting
        self.horizon = horizon
        self.c_switch = c_switch
        self._obs_probs = OBS_PROBS[setting]       # shape (4, 3, 3)
        self._rng = np.random.default_rng(seed)
        self._state: Optional[Tuple[int, int, int]] = None

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def reset(self) -> Tuple[int, int, int]:
        """Reset to initial state s0 = (Start=0, baseline=0, t=0)."""
        self._state = (0, 0, 0)
        return self._state

    def step(
        self, action: int
    ) -> Tuple[Tuple[int, int, int], float, bool, dict]:
        """
        Apply *action* from the current state.

        Returns
        -------
        next_state : (site_idx, ps_idx, t+1)
        reward     : g(obs) - c_switch * is_switch
        done       : True when t+1 >= horizon
        info       : {'observation': str, 'patient_state': str}
        """
        if self._state is None:
            raise RuntimeError("Call reset() before step().")
        if action not in range(N_ACTIONS):
            raise ValueError(f"action must be in 0..{N_ACTIONS - 1}, got {action}")

        site_idx, ps_idx, t = self._state
        action_site_idx = action + 1   # S1→1, S2→2, S3→3, S4→4

        # Same-site and switch-cost logic
        # The Start site (site_idx=0) never counts as "same" or incurs a switch cost.
        is_same   = (site_idx == action_site_idx) and (site_idx != 0)
        is_switch = (not is_same) and (site_idx != 0)
        switch_cost = self.c_switch if is_switch else 0.0

        # Sample next patient state
        t_probs = PATIENT_TRANSITION[(ps_idx, is_same)]
        new_ps_idx = int(self._rng.choice(N_PATIENT_STATES, p=t_probs))

        # Sample observation (from CURRENT patient state + chosen action site)
        obs_probs = self._obs_probs[action, ps_idx]
        obs_idx = int(self._rng.choice(N_OBS, p=obs_probs))
        obs = OBSERVATIONS[obs_idx]

        reward = float(REWARD_MAP[obs]) - switch_cost
        next_state = (action_site_idx, new_ps_idx, t + 1)
        done = (t + 1 >= self.horizon)

        self._state = next_state
        info = {
            'observation':   obs,
            'patient_state': PATIENT_STATES[new_ps_idx],
        }
        return next_state, reward, done, info

    # ------------------------------------------------------------------
    # Space descriptors
    # ------------------------------------------------------------------

    def state_space(self) -> List[Tuple[int, int, int]]:
        """All valid (site_idx, ps_idx, t) tuples including terminal t=horizon."""
        return [
            (site, ps, t)
            for site in range(N_SITES_STATE)
            for ps   in range(N_PATIENT_STATES)
            for t    in range(self.horizon + 1)
        ]

    def action_space(self) -> List[int]:
        """Valid action indices [0, 1, 2, 3]."""
        return list(range(N_ACTIONS))

    def is_terminal(self, state: Tuple[int, int, int]) -> bool:
        """True when t == horizon (no further transitions possible)."""
        return state[2] >= self.horizon

    @staticmethod
    def decode_state(state: Tuple[int, int, int]) -> str:
        """Human-readable representation of a state tuple.

        Examples
        --------
        >>> StimulationEnv.decode_state((0, 0, 0))
        'Start/baseline/t=0'
        >>> StimulationEnv.decode_state((1, 1, 3))
        'S1/receptive/t=3'
        """
        site_idx, ps_idx, t = state
        site_name = 'Start' if site_idx == 0 else SITES[site_idx - 1]
        return f"{site_name}/{PATIENT_STATES[ps_idx]}/t={t}"

    # ------------------------------------------------------------------
    # Model access for Value Iteration
    # ------------------------------------------------------------------

    def get_transition_probs(
        self,
        state: Tuple[int, int, int],
        action: int,
    ) -> List[Tuple[Tuple[int, int, int], float, float]]:
        """
        Full transition distribution from *state* under *action*.

        Returns a list of (next_state, reward, probability) tuples.
        The patient-state transition and observation model are independent
        given (state, action), yielding up to 3×3 = 9 outcome tuples.
        """
        site_idx, ps_idx, t = state
        action_site_idx = action + 1

        is_same   = (site_idx == action_site_idx) and (site_idx != 0)
        is_switch = (not is_same) and (site_idx != 0)
        switch_cost = self.c_switch if is_switch else 0.0

        t_probs   = PATIENT_TRANSITION[(ps_idx, is_same)]
        obs_probs = self._obs_probs[action, ps_idx]

        outcomes = []
        for new_ps_idx in range(N_PATIENT_STATES):
            ps_prob = float(t_probs[new_ps_idx])
            if ps_prob == 0.0:
                continue
            for obs_idx in range(N_OBS):
                obs_prob = float(obs_probs[obs_idx])
                if obs_prob == 0.0:
                    continue
                reward     = float(REWARD_MAP[OBSERVATIONS[obs_idx]]) - switch_cost
                next_state = (action_site_idx, new_ps_idx, t + 1)
                outcomes.append((next_state, reward, ps_prob * obs_prob))
        return outcomes
