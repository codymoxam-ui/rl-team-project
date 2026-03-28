"""
Phase 2 tests: StimulationEnv MDP environment.

Sections
--------
1.  Initialization & parameter validation
2.  reset()
3.  action_space() / state_space()
4.  is_terminal()
5.  Probability table sanity (PATIENT_TRANSITION, OBS_PROBS)
6.  step() — return types, state transitions, rewards
7.  Switch-cost logic
8.  Episode termination
9.  get_transition_probs() — correctness and probability sums
10. Seeded reproducibility
11. Multi-setting smoke test
"""
import pytest
import numpy as np

from src.env.stimulation_env import (
    N_ACTIONS,
    N_OBS,
    N_PATIENT_STATES,
    N_SITES_STATE,
    OBSERVATIONS,
    OBS_PROBS,
    PATIENT_STATES,
    PATIENT_TRANSITION,
    REWARD_MAP,
    SITES,
    VALID_SETTINGS,
    StimulationEnv,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_env(**kwargs) -> StimulationEnv:
    defaults = dict(setting='high', horizon=10, c_switch=0.0, seed=42)
    defaults.update(kwargs)
    return StimulationEnv(**defaults)


def run_episode(env: StimulationEnv, policy=None) -> float:
    """Run one episode; return total return."""
    state = env.reset()
    total = 0.0
    done = False
    rng = np.random.default_rng(0)
    while not done:
        action = policy(state) if policy else int(rng.integers(N_ACTIONS))
        _, reward, done, _ = env.step(action)
        total += reward
    return total


# ===========================================================================
# 1. Initialization & parameter validation
# ===========================================================================

def test_default_init():
    env = StimulationEnv()
    assert env.setting == 'high'
    assert env.horizon == 10
    assert env.c_switch == 0.0


@pytest.mark.parametrize("setting", VALID_SETTINGS)
def test_valid_settings(setting):
    env = StimulationEnv(setting=setting)
    assert env.setting == setting


def test_invalid_setting_raises():
    with pytest.raises(ValueError, match="setting"):
        StimulationEnv(setting='extreme')


def test_invalid_horizon_raises():
    with pytest.raises(ValueError, match="horizon"):
        StimulationEnv(horizon=0)


def test_invalid_c_switch_raises():
    with pytest.raises(ValueError, match="c_switch"):
        StimulationEnv(c_switch=-0.1)


@pytest.mark.parametrize("horizon", [1, 5, 10, 20])
def test_various_horizons(horizon):
    env = StimulationEnv(horizon=horizon)
    assert env.horizon == horizon


@pytest.mark.parametrize("c_switch", [0.0, 0.1, 0.25])
def test_valid_c_switch_values(c_switch):
    env = StimulationEnv(c_switch=c_switch)
    assert env.c_switch == c_switch


# ===========================================================================
# 2. reset()
# ===========================================================================

def test_reset_returns_initial_state():
    env = make_env()
    state = env.reset()
    assert state == (0, 0, 0), "Initial state must be (Start=0, baseline=0, t=0)"


def test_reset_returns_tuple_of_ints():
    env = make_env()
    state = env.reset()
    assert isinstance(state, tuple)
    assert all(isinstance(x, int) for x in state)


def test_reset_is_repeatable():
    env = make_env()
    s1 = env.reset()
    env.step(0)
    s2 = env.reset()
    assert s1 == s2


# ===========================================================================
# 3. action_space() / state_space()
# ===========================================================================

def test_action_space_values():
    env = make_env()
    assert env.action_space() == [0, 1, 2, 3]


def test_action_space_length():
    env = make_env()
    assert len(env.action_space()) == N_ACTIONS


@pytest.mark.parametrize("horizon", [5, 10])
def test_state_space_size(horizon):
    env = make_env(horizon=horizon)
    expected = N_SITES_STATE * N_PATIENT_STATES * (horizon + 1)
    assert len(env.state_space()) == expected


def test_state_space_contains_initial_state():
    env = make_env()
    assert (0, 0, 0) in env.state_space()


def test_state_space_no_duplicates():
    env = make_env(horizon=5)
    states = env.state_space()
    assert len(states) == len(set(states))


def test_state_space_all_indices_valid():
    env = make_env(horizon=5)
    for site, ps, t in env.state_space():
        assert 0 <= site < N_SITES_STATE
        assert 0 <= ps < N_PATIENT_STATES
        assert 0 <= t <= env.horizon


# ===========================================================================
# 4. is_terminal()
# ===========================================================================

def test_terminal_at_horizon():
    env = make_env(horizon=10)
    assert env.is_terminal((1, 0, 10))


def test_not_terminal_before_horizon():
    env = make_env(horizon=10)
    assert not env.is_terminal((1, 0, 9))
    assert not env.is_terminal((0, 0, 0))


def test_terminal_beyond_horizon():
    env = make_env(horizon=5)
    assert env.is_terminal((1, 0, 6))


# ===========================================================================
# 5. Probability table sanity
# ===========================================================================

@pytest.mark.parametrize("key", list(PATIENT_TRANSITION.keys()))
def test_patient_transition_sums_to_one(key):
    probs = PATIENT_TRANSITION[key]
    assert abs(probs.sum() - 1.0) < 1e-9, f"PATIENT_TRANSITION[{key}] sums to {probs.sum()}"


@pytest.mark.parametrize("key", list(PATIENT_TRANSITION.keys()))
def test_patient_transition_all_non_negative(key):
    assert np.all(PATIENT_TRANSITION[key] >= 0)


@pytest.mark.parametrize("setting", VALID_SETTINGS)
@pytest.mark.parametrize("site_idx", range(4))
@pytest.mark.parametrize("ps_idx", range(3))
def test_obs_probs_sum_to_one(setting, site_idx, ps_idx):
    probs = OBS_PROBS[setting][site_idx, ps_idx]
    assert abs(probs.sum() - 1.0) < 1e-9, (
        f"OBS_PROBS[{setting}][{site_idx},{ps_idx}] sums to {probs.sum()}"
    )


@pytest.mark.parametrize("setting", VALID_SETTINGS)
def test_obs_probs_all_non_negative(setting):
    assert np.all(OBS_PROBS[setting] >= 0)


# ===========================================================================
# 6. step() — return types, state transitions, rewards
# ===========================================================================

def test_step_before_reset_raises():
    env = StimulationEnv()
    with pytest.raises(RuntimeError):
        env.step(0)


def test_step_invalid_action_raises():
    env = make_env()
    env.reset()
    with pytest.raises(ValueError):
        env.step(4)
    with pytest.raises(ValueError):
        env.step(-1)


def test_step_return_types():
    env = make_env()
    env.reset()
    next_state, reward, done, info = env.step(0)
    assert isinstance(next_state, tuple) and len(next_state) == 3
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert isinstance(info, dict)


def test_step_next_state_site_matches_action():
    env = make_env(seed=0)
    env.reset()
    for action in range(N_ACTIONS):
        env.reset()
        next_state, _, _, _ = env.step(action)
        assert next_state[0] == action + 1, (
            f"action {action} should move to site_idx {action+1}, got {next_state[0]}"
        )


def test_step_time_increments():
    env = make_env()
    env.reset()
    for expected_t in range(1, 5):
        ns, _, _, _ = env.step(0)
        assert ns[2] == expected_t


def test_step_patient_state_in_valid_range():
    env = make_env(seed=7)
    env.reset()
    for _ in range(10):
        ns, _, done, _ = env.step(0)
        assert 0 <= ns[1] < N_PATIENT_STATES
        if done:
            break


def test_step_info_has_required_keys():
    env = make_env()
    env.reset()
    _, _, _, info = env.step(0)
    assert 'observation' in info
    assert 'patient_state' in info


def test_step_observation_is_valid_string():
    env = make_env()
    env.reset()
    _, _, _, info = env.step(0)
    assert info['observation'] in OBSERVATIONS


def test_step_patient_state_is_valid_string():
    env = make_env()
    env.reset()
    _, _, _, info = env.step(0)
    assert info['patient_state'] in PATIENT_STATES


def test_step_reward_matches_observation_no_switch():
    """With c_switch=0 and seeded env, verify reward == REWARD_MAP[obs]."""
    env = make_env(c_switch=0.0, seed=99)
    env.reset()
    for _ in range(10):
        _, reward, done, info = env.step(0)  # keep same site, no switch
        expected = float(REWARD_MAP[info['observation']])
        assert reward == expected
        if done:
            break


def test_step_state_advances_internally():
    env = make_env()
    env.reset()
    ns1, _, _, _ = env.step(0)
    ns2, _, _, _ = env.step(1)
    assert ns2[2] == 2  # t should be 2 after two steps


# ===========================================================================
# 7. Switch-cost logic
# ===========================================================================

def test_no_switch_cost_from_start():
    """First action from Start should never incur switch cost."""
    env = make_env(c_switch=0.25, seed=0)
    env_no_cost = make_env(c_switch=0.0, seed=0)

    env.reset()
    env_no_cost.reset()

    # Same seed → same obs → reward difference must be 0 (no switch cost)
    _, r_cost, _, info = env.step(1)
    _, r_no,   _, _    = env_no_cost.step(1)
    assert r_cost == r_no, "No switch cost should apply from Start state"


def test_switch_cost_applied_on_site_change():
    """Switching from S1 to S2 incurs c_switch."""
    c = 0.25
    env_cost = make_env(c_switch=c, seed=5)
    env_free = make_env(c_switch=0.0, seed=5)

    # Advance both to same non-Start state at S1
    for e in (env_cost, env_free):
        e.reset()
        e.step(0)   # go to S1

    # Now switch to S2
    _, r_cost, _, info_cost = env_cost.step(1)
    _, r_free, _, info_free = env_free.step(1)

    # Same obs (same seed), so reward difference = c_switch
    assert info_cost['observation'] == info_free['observation']
    assert abs((r_free - r_cost) - c) < 1e-9


def test_no_switch_cost_same_site():
    """Staying on the same site never incurs switch cost."""
    c = 0.25
    env_cost = make_env(c_switch=c, seed=3)
    env_free = make_env(c_switch=0.0, seed=3)

    for e in (env_cost, env_free):
        e.reset()
        e.step(0)   # go to S1

    # Stay on S1
    _, r_cost, _, info_cost = env_cost.step(0)
    _, r_free, _, info_free = env_free.step(0)

    assert info_cost['observation'] == info_free['observation']
    assert r_cost == r_free, "No switch cost should apply when staying on same site"


# ===========================================================================
# 8. Episode termination
# ===========================================================================

@pytest.mark.parametrize("horizon", [1, 5, 10])
def test_episode_terminates_exactly_at_horizon(horizon):
    env = make_env(horizon=horizon)
    env.reset()
    done = False
    steps = 0
    while not done:
        _, _, done, _ = env.step(steps % N_ACTIONS)
        steps += 1
    assert steps == horizon, f"Episode should last exactly {horizon} steps, took {steps}"


def test_done_flag_false_before_terminal():
    env = make_env(horizon=5)
    env.reset()
    for _ in range(4):
        _, _, done, _ = env.step(0)
        assert not done


def test_done_flag_true_at_last_step():
    env = make_env(horizon=5)
    env.reset()
    done = False
    for _ in range(5):
        _, _, done, _ = env.step(0)
    assert done


def test_multiple_episodes_consistent_length():
    env = make_env(horizon=5)
    for _ in range(10):
        env.reset()
        steps = 0
        done = False
        while not done:
            _, _, done, _ = env.step(steps % N_ACTIONS)
            steps += 1
        assert steps == 5


# ===========================================================================
# 9. get_transition_probs()
# ===========================================================================

@pytest.mark.parametrize("setting", VALID_SETTINGS)
@pytest.mark.parametrize("action", range(N_ACTIONS))
def test_transition_probs_sum_to_one_from_initial(setting, action):
    env = make_env(setting=setting)
    outcomes = env.get_transition_probs((0, 0, 0), action)
    total = sum(prob for _, _, prob in outcomes)
    assert abs(total - 1.0) < 1e-9, f"probs sum to {total} for setting={setting}, action={action}"


@pytest.mark.parametrize("state", [(1, 0, 1), (2, 1, 3), (3, 2, 7)])
@pytest.mark.parametrize("action", range(N_ACTIONS))
def test_transition_probs_sum_to_one_various_states(state, action):
    env = make_env(horizon=10)
    outcomes = env.get_transition_probs(state, action)
    total = sum(prob for _, _, prob in outcomes)
    assert abs(total - 1.0) < 1e-9


def test_transition_probs_all_non_negative():
    env = make_env()
    for state in [(0, 0, 0), (1, 1, 2), (4, 2, 5)]:
        for action in range(N_ACTIONS):
            for _, _, prob in env.get_transition_probs(state, action):
                assert prob >= 0


def test_transition_probs_next_site_matches_action():
    env = make_env()
    for action in range(N_ACTIONS):
        outcomes = env.get_transition_probs((0, 0, 0), action)
        for next_state, _, _ in outcomes:
            assert next_state[0] == action + 1


def test_transition_probs_time_increments():
    env = make_env()
    outcomes = env.get_transition_probs((1, 0, 3), 0)
    for next_state, _, _ in outcomes:
        assert next_state[2] == 4


def test_transition_probs_reward_range():
    """All rewards must be in [-1-c_switch, +1]."""
    c = 0.25
    env = make_env(c_switch=c)
    outcomes = env.get_transition_probs((1, 0, 1), 1)  # switching site
    for _, reward, _ in outcomes:
        assert -1 - c - 1e-9 <= reward <= 1 + 1e-9


def test_transition_probs_no_switch_cost_from_start():
    """From Start, no switch cost regardless of c_switch."""
    c = 0.25
    env_cost = make_env(c_switch=c)
    env_free = make_env(c_switch=0.0)
    for action in range(N_ACTIONS):
        out_cost = {(ns, r): p for ns, r, p in env_cost.get_transition_probs((0, 0, 0), action)}
        out_free = {(ns, r): p for ns, r, p in env_free.get_transition_probs((0, 0, 0), action)}
        assert out_cost == out_free, "No switch cost from Start state"


def test_transition_probs_switch_cost_applied():
    """From S1 (site_idx=1), switching to S2 (action=1) should offset rewards by c_switch."""
    c = 0.1
    env_cost = make_env(c_switch=c)
    env_free = make_env(c_switch=0.0)
    state = (1, 0, 1)   # at S1
    action = 1           # switch to S2

    rewards_cost = sorted(r for _, r, _ in env_cost.get_transition_probs(state, action))
    rewards_free = sorted(r for _, r, _ in env_free.get_transition_probs(state, action))
    for rc, rf in zip(rewards_cost, rewards_free):
        assert abs((rf - rc) - c) < 1e-9


def test_transition_probs_consistent_with_step_distribution():
    """
    Empirically verify get_transition_probs matches step() distribution.
    Run many episodes from a fixed state and compare observed frequencies to
    the theoretical probabilities.
    """
    env = make_env(c_switch=0.0, seed=0)
    state = (0, 0, 0)
    action = 0           # go to S1

    # Theoretical distribution
    theory = {}
    for ns, r, p in env.get_transition_probs(state, action):
        key = (ns, r)
        theory[key] = theory.get(key, 0) + p

    # Empirical
    n = 20_000
    counts: dict = {}
    rng = np.random.default_rng(42)
    obs_env = StimulationEnv(setting='high', horizon=10, c_switch=0.0, seed=None)
    obs_env._rng = rng
    for _ in range(n):
        obs_env._state = state
        ns, r, _, _ = obs_env.step(action)
        key = (ns, r)
        counts[key] = counts.get(key, 0) + 1

    for key, expected_p in theory.items():
        observed_p = counts.get(key, 0) / n
        assert abs(observed_p - expected_p) < 0.02, (
            f"Empirical p={observed_p:.4f} vs theoretical p={expected_p:.4f} for {key}"
        )


# ===========================================================================
# 10. decode_state() human-readable helper
# ===========================================================================

def test_decode_state_initial():
    assert StimulationEnv.decode_state((0, 0, 0)) == 'Start/baseline/t=0'


def test_decode_state_site_mapping():
    for action in range(N_ACTIONS):
        label = StimulationEnv.decode_state((action + 1, 0, 1))
        assert label.startswith(SITES[action]), f"Expected {SITES[action]}, got {label}"


def test_decode_state_patient_state_mapping():
    for ps_idx, ps_name in enumerate(PATIENT_STATES):
        label = StimulationEnv.decode_state((1, ps_idx, 2))
        assert ps_name in label


def test_decode_state_timestep():
    assert StimulationEnv.decode_state((2, 1, 7)) == 'S2/receptive/t=7'


def test_decode_state_terminal():
    env = make_env(horizon=10)
    env.reset()
    for _ in range(10):
        ns, _, done, _ = env.step(0)
        if done:
            label = StimulationEnv.decode_state(ns)
            assert 't=10' in label


# ===========================================================================
# 11. Seeded reproducibility
# ===========================================================================

def test_same_seed_same_rollout():
    def collect(seed):
        env = make_env(seed=seed)
        env.reset()
        rng = np.random.default_rng(0)
        rewards = []
        for _ in range(env.horizon):
            _, r, done, _ = env.step(int(rng.integers(N_ACTIONS)))
            rewards.append(r)
            if done:
                break
        return rewards

    assert collect(42) == collect(42)


def test_same_seed_same_trajectory():
    def get_trajectory(seed):
        env = make_env(seed=seed)
        env.reset()
        traj = []
        for a in [0, 1, 0, 2, 1, 0, 3, 1, 2, 0]:
            ns, r, done, _ = env.step(a)
            traj.append((ns, r))
            if done:
                break
        return traj

    assert get_trajectory(123) == get_trajectory(123)


def test_different_seeds_different_trajectories():
    def get_rewards(seed):
        env = make_env(seed=seed)
        env.reset()
        rewards = []
        rng = np.random.default_rng(seed)
        for _ in range(env.horizon):
            _, r, done, _ = env.step(int(rng.integers(N_ACTIONS)))
            rewards.append(r)
            if done:
                break
        return rewards

    # Very unlikely to be identical across many steps
    assert get_rewards(0) != get_rewards(999)


# ===========================================================================
# 11. Multi-setting smoke test
# ===========================================================================

@pytest.mark.parametrize("setting", VALID_SETTINGS)
@pytest.mark.parametrize("horizon", [5, 10])
@pytest.mark.parametrize("c_switch", [0.0, 0.1, 0.25])
def test_full_episode_completes(setting, horizon, c_switch):
    env = StimulationEnv(setting=setting, horizon=horizon, c_switch=c_switch, seed=0)
    total = run_episode(env)
    # Return must be in [-horizon*(1+c_switch), +horizon]
    assert -(horizon * (1 + c_switch)) - 1e-9 <= total <= horizon + 1e-9


@pytest.mark.parametrize("setting", VALID_SETTINGS)
def test_random_policy_return_finite(setting):
    env = make_env(setting=setting)
    for _ in range(5):
        total = run_episode(env)
        assert np.isfinite(total)
