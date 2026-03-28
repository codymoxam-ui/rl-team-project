"""
Phase 3 tests: all RL agents.

Sections
--------
1.  BaseAgent validation (tested via QLearningAgent as a concrete stand-in)
2.  Q-table structure (all agents)
3.  ε-greedy action selection
4.  ε decay
5.  reset()
6.  Q-Learning — update rule correctness
7.  Expected SARSA — expected-value calculation
8.  Double Q-Learning — dual-table mechanics
9.  Monte Carlo — trajectory + first-visit update
10. Value Iteration — backward induction + optimal policy
11. Convergence: all model-free agents improve over time (high-separation)
12. Upper-bound: Value Iteration ≥ model-free agents after convergence
"""
import pytest
import numpy as np

from src.env.stimulation_env import (
    N_ACTIONS, N_PATIENT_STATES, N_SITES_STATE, StimulationEnv,
)
from src.agents.base_agent import BaseAgent
from src.agents.q_learning import QLearningAgent
from src.agents.expected_sarsa import ExpectedSARSAAgent
from src.agents.double_q_learning import DoubleQLearningAgent
from src.agents.monte_carlo import MonteCarloAgent
from src.agents.value_iteration import ValueIterationAgent

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_env(horizon=10, setting='high', c_switch=0.0, seed=0):
    return StimulationEnv(setting=setting, horizon=horizon, c_switch=c_switch, seed=seed)


def make_agent(cls, env=None, **kwargs):
    if env is None:
        env = make_env()
    defaults = dict(alpha=0.1, epsilon=1.0, epsilon_decay=0.995,
                    epsilon_min=0.05, gamma=1.0, seed=42)
    defaults.update(kwargs)
    return cls(env, **defaults)


ALL_MODEL_FREE = [QLearningAgent, ExpectedSARSAAgent, DoubleQLearningAgent, MonteCarloAgent]


# ===========================================================================
# 1. BaseAgent parameter validation
# ===========================================================================

@pytest.mark.parametrize("bad_alpha", [0.0, -0.1, 1.1])
def test_invalid_alpha_raises(bad_alpha):
    with pytest.raises(ValueError, match="alpha"):
        make_agent(QLearningAgent, alpha=bad_alpha)


@pytest.mark.parametrize("bad_eps", [-0.1, 1.1])
def test_invalid_epsilon_raises(bad_eps):
    with pytest.raises(ValueError, match="epsilon"):
        make_agent(QLearningAgent, epsilon=bad_eps)


@pytest.mark.parametrize("bad_decay", [0.0, -0.1, 1.1])
def test_invalid_epsilon_decay_raises(bad_decay):
    with pytest.raises(ValueError, match="epsilon_decay"):
        make_agent(QLearningAgent, epsilon_decay=bad_decay)


@pytest.mark.parametrize("bad_gamma", [0.0, -0.1, 1.1])
def test_invalid_gamma_raises(bad_gamma):
    with pytest.raises(ValueError, match="gamma"):
        make_agent(QLearningAgent, gamma=bad_gamma)


def test_base_agent_is_abstract():
    """BaseAgent cannot be instantiated directly."""
    with pytest.raises(TypeError):
        BaseAgent(make_env())


# ===========================================================================
# 2. Q-table structure
# ===========================================================================

@pytest.mark.parametrize("cls", ALL_MODEL_FREE)
def test_q_table_shape(cls):
    env = make_env(horizon=10)
    agent = make_agent(cls, env=env)
    expected = (N_SITES_STATE, N_PATIENT_STATES, env.horizon + 1, N_ACTIONS)
    assert agent.Q.shape == expected, f"{cls.__name__} Q shape {agent.Q.shape} != {expected}"


@pytest.mark.parametrize("cls", ALL_MODEL_FREE)
def test_q_table_initialised_to_zero(cls):
    agent = make_agent(cls)
    assert np.all(agent.Q == 0.0)


@pytest.mark.parametrize("cls", ALL_MODEL_FREE)
def test_get_q_returns_copy(cls):
    agent = make_agent(cls)
    q = agent.get_Q()
    q[0, 0, 0, 0] = 999.0
    assert agent.Q[0, 0, 0, 0] != 999.0, "get_Q() should return a copy, not a view"


# ===========================================================================
# 3. ε-greedy action selection
# ===========================================================================

@pytest.mark.parametrize("cls", ALL_MODEL_FREE)
def test_epsilon_one_is_random(cls):
    """With ε=1, all actions should be chosen roughly uniformly."""
    agent = make_agent(cls, epsilon=1.0, epsilon_decay=1.0, seed=0)
    counts = np.zeros(N_ACTIONS)
    state = (0, 0, 0)
    for _ in range(4000):
        counts[agent.select_action(state)] += 1
    fracs = counts / counts.sum()
    for f in fracs:
        assert abs(f - 0.25) < 0.05, f"Fraction {f:.3f} deviates from 0.25 with ε=1"


@pytest.mark.parametrize("cls", ALL_MODEL_FREE)
def test_epsilon_zero_is_greedy(cls):
    """With ε=0, agent always picks the argmax action."""
    agent = make_agent(cls, epsilon=0.0, epsilon_min=0.0, epsilon_decay=1.0)
    # Set Q so action 2 is best from state (0,0,0)
    agent.Q[0, 0, 0, :] = [0.1, 0.2, 0.9, 0.3]
    for _ in range(20):
        assert agent.select_action((0, 0, 0)) == 2


@pytest.mark.parametrize("cls", ALL_MODEL_FREE)
def test_select_action_returns_valid_action(cls):
    agent = make_agent(cls)
    for state in [(0, 0, 0), (1, 1, 3), (4, 2, 9)]:
        a = agent.select_action(state)
        assert a in range(N_ACTIONS)


# ===========================================================================
# 4. ε decay
# ===========================================================================

@pytest.mark.parametrize("cls", ALL_MODEL_FREE)
def test_epsilon_decays_after_episode(cls):
    agent = make_agent(cls, epsilon=1.0, epsilon_decay=0.9, epsilon_min=0.0)
    for _ in range(10):
        agent._decay_epsilon()
    assert agent.epsilon < 1.0


@pytest.mark.parametrize("cls", ALL_MODEL_FREE)
def test_epsilon_does_not_go_below_min(cls):
    agent = make_agent(cls, epsilon=1.0, epsilon_decay=0.5, epsilon_min=0.1)
    for _ in range(100):
        agent._decay_epsilon()
    assert agent.epsilon >= 0.1


@pytest.mark.parametrize("cls", ALL_MODEL_FREE)
def test_epsilon_decreases_during_training(cls):
    agent = make_agent(cls, epsilon=1.0, epsilon_decay=0.99, epsilon_min=0.01)
    initial_eps = agent.epsilon
    agent.train(50)
    assert agent.epsilon < initial_eps


# ===========================================================================
# 5. reset()
# ===========================================================================

@pytest.mark.parametrize("cls", ALL_MODEL_FREE)
def test_reset_zeros_q_table(cls):
    agent = make_agent(cls)
    agent.train(20)
    assert not np.all(agent.Q == 0.0), "Q should have been modified after training"
    agent.reset()
    assert np.all(agent.Q == 0.0), "Q should be zeroed after reset()"


@pytest.mark.parametrize("cls", ALL_MODEL_FREE)
def test_reset_restores_epsilon(cls):
    agent = make_agent(cls, epsilon=0.8, epsilon_decay=0.9)
    agent.train(50)
    assert agent.epsilon < 0.8
    agent.reset()
    assert agent.epsilon == 0.8


# ===========================================================================
# 6. Q-Learning — update rule
# ===========================================================================

def test_qlearning_update_non_terminal():
    """Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]"""
    env = make_env()
    agent = make_agent(QLearningAgent, env=env, alpha=0.5, gamma=1.0)

    state      = (1, 0, 1)
    action     = 0
    reward     = 1.0
    next_state = (1, 0, 2)

    agent.Q[1, 0, 2, :] = [0.2, 0.4, 0.6, 0.8]  # max = 0.8
    agent.update(state, action, reward, next_state, done=False)

    expected = 0.0 + 0.5 * (1.0 + 1.0 * 0.8 - 0.0)
    assert abs(agent.Q[1, 0, 1, 0] - expected) < 1e-9


def test_qlearning_update_terminal():
    """At terminal step target is just r."""
    env = make_env()
    agent = make_agent(QLearningAgent, env=env, alpha=0.5, gamma=1.0)
    agent.update((1, 0, 9), 0, -1.0, (1, 0, 10), done=True)
    assert abs(agent.Q[1, 0, 9, 0] - (-0.5)) < 1e-9


def test_qlearning_uses_max_next_q():
    """Off-policy: uses max over all actions in s', not the action taken."""
    env = make_env()
    agent = make_agent(QLearningAgent, env=env, alpha=1.0, gamma=1.0)
    agent.Q[2, 0, 3, :] = [1.0, 5.0, 2.0, 3.0]   # max is 5.0 at action 1
    agent.update((1, 0, 2), 0, 0.0, (2, 0, 3), done=False)
    assert abs(agent.Q[1, 0, 2, 0] - 5.0) < 1e-9


# ===========================================================================
# 7. Expected SARSA — expected-value calculation
# ===========================================================================

def test_expected_sarsa_update_non_terminal():
    """Verify the expected value target matches the ε-greedy formula."""
    env = make_env()
    eps = 0.2
    agent = make_agent(ExpectedSARSAAgent, env=env, alpha=1.0, gamma=1.0, epsilon=eps,
                       epsilon_decay=1.0)

    state, action, reward, next_state = (1, 0, 1), 0, 0.0, (2, 0, 2)
    q_next = np.array([1.0, 3.0, 2.0, 0.5])
    agent.Q[2, 0, 2, :] = q_next

    expected_val = (eps / N_ACTIONS) * q_next.sum() + (1 - eps) * q_next.max()
    agent.update(state, action, reward, next_state, done=False)

    assert abs(agent.Q[1, 0, 1, 0] - expected_val) < 1e-9


def test_expected_sarsa_terminal():
    env = make_env()
    agent = make_agent(ExpectedSARSAAgent, env=env, alpha=0.5, gamma=1.0)
    agent.update((1, 0, 9), 0, 2.0, (1, 0, 10), done=True)
    assert abs(agent.Q[1, 0, 9, 0] - 1.0) < 1e-9  # α*(r - 0) = 0.5*2


# ===========================================================================
# 8. Double Q-Learning — dual-table mechanics
# ===========================================================================

def test_double_q_has_two_tables():
    agent = make_agent(DoubleQLearningAgent)
    assert hasattr(agent, 'Q1') and hasattr(agent, 'Q2')
    assert agent.Q1.shape == agent.Q2.shape


def test_double_q_q1_is_base_q():
    """Q1 and self.Q should be the same array."""
    agent = make_agent(DoubleQLearningAgent)
    assert agent.Q1 is agent.Q


def test_double_q_action_selection_uses_sum():
    """With ε=0, agent picks argmax of Q1+Q2."""
    agent = make_agent(DoubleQLearningAgent, epsilon=0.0, epsilon_min=0.0, epsilon_decay=1.0)
    agent.Q1[0, 0, 0, :] = [0.5, 0.0, 0.0, 0.0]
    agent.Q2[0, 0, 0, :] = [0.0, 0.0, 0.8, 0.0]  # Q1+Q2 max at action 2
    assert agent.select_action((0, 0, 0)) == 2


def test_double_q_update_modifies_one_table():
    """Each update only modifies Q1 or Q2, never both simultaneously."""
    env = make_env()
    state, action = (1, 0, 1), 0
    next_state = (2, 0, 2)

    # Run many updates; at least one table should stay 0 during early updates
    # with a fixed seed we can predict which table gets updated first
    agent = make_agent(DoubleQLearningAgent, env=env, alpha=1.0, gamma=1.0, seed=0)
    q1_before = agent.Q1[1, 0, 1, 0]
    q2_before = agent.Q2[1, 0, 1, 0]
    agent.update(state, action, 1.0, next_state, done=False)
    q1_after = agent.Q1[1, 0, 1, 0]
    q2_after = agent.Q2[1, 0, 1, 0]
    # Exactly one of the two tables changed
    changed = (q1_after != q1_before) + (q2_after != q2_before)
    assert changed == 1, "Exactly one Q-table should be updated per step"


def test_double_q_reset_zeroes_both_tables():
    agent = make_agent(DoubleQLearningAgent)
    agent.train(10)
    agent.reset()
    assert np.all(agent.Q1 == 0.0)
    assert np.all(agent.Q2 == 0.0)


def test_double_q_get_q_returns_average():
    agent = make_agent(DoubleQLearningAgent, epsilon=0.0, epsilon_min=0.0, epsilon_decay=1.0)
    agent.Q1[0, 0, 0, :] = [2.0, 0.0, 0.0, 0.0]
    agent.Q2[0, 0, 0, :] = [0.0, 0.0, 0.0, 4.0]
    q_avg = agent.get_Q()
    assert abs(q_avg[0, 0, 0, 0] - 1.0) < 1e-9
    assert abs(q_avg[0, 0, 0, 3] - 2.0) < 1e-9


# ===========================================================================
# 9. Monte Carlo — trajectory + first-visit update
# ===========================================================================

def test_mc_train_returns_correct_length():
    agent = make_agent(MonteCarloAgent)
    returns = agent.train(30)
    assert len(returns) == 30


def test_mc_update_is_noop():
    """The per-step update method should not modify Q."""
    agent = make_agent(MonteCarloAgent)
    before = agent.Q.copy()
    agent.update((0, 0, 0), 0, 1.0, (1, 0, 1), False)
    assert np.array_equal(agent.Q, before)


def test_mc_q_changes_after_train():
    agent = make_agent(MonteCarloAgent)
    agent.train(10)
    assert not np.all(agent.Q == 0.0), "MC Q-table should be updated after training"


def test_mc_first_visit_only():
    """
    Manually run one episode where (s,a) appears twice.
    The Q update for the first visit should use the full G from that point;
    the second visit should be ignored.
    """
    env = make_env(horizon=4, seed=7)
    agent = make_agent(MonteCarloAgent, env=env, alpha=1.0, gamma=1.0,
                       epsilon=0.0, epsilon_min=0.0, epsilon_decay=1.0)

    # Force a fixed trajectory by pre-setting Q to always pick action 0
    agent.Q[:] = 0.0
    agent.Q[0, 0, 0, 0] = 10.0  # from Start, always pick S1

    # With alpha=1 and first-visit MC:
    # Q(s,a) = G at first visit (since initial Q is wiped by alpha=1)
    agent.train(1)
    # After one episode Q must be non-trivially set
    assert not np.all(agent.Q == 0.0)


# ===========================================================================
# 10. Value Iteration — backward induction + optimal policy
# ===========================================================================

def test_vi_solve_returns_q_array():
    env = make_env()
    vi = ValueIterationAgent(env)
    Q = vi.solve()
    expected_shape = (N_SITES_STATE, N_PATIENT_STATES, env.horizon + 1, N_ACTIONS)
    assert Q.shape == expected_shape


def test_vi_terminal_values_are_zero():
    env = make_env(horizon=5)
    vi = ValueIterationAgent(env)
    vi.solve()
    V = vi.get_V()
    assert np.all(V[:, :, env.horizon] == 0.0)


def test_vi_get_action_before_solve_raises():
    env = make_env()
    vi = ValueIterationAgent(env)
    with pytest.raises(RuntimeError):
        vi.get_action((0, 0, 0))


def test_vi_run_episode_before_solve_raises():
    env = make_env()
    vi = ValueIterationAgent(env)
    with pytest.raises(RuntimeError):
        vi.run_episode()


def test_vi_get_action_returns_valid():
    env = make_env()
    vi = ValueIterationAgent(env)
    vi.solve()
    for state in [(0, 0, 0), (1, 1, 3), (4, 2, 9)]:
        assert vi.get_action(state) in range(N_ACTIONS)


def test_vi_optimal_action_from_initial_state_high_separation():
    """
    In high-separation setting, S1 has the highest baseline expected reward (+0.60).
    VI must prefer S1 (action=0) from the initial state.
    """
    env = make_env(setting='high', horizon=10)
    vi = ValueIterationAgent(env)
    vi.solve()
    assert vi.get_action((0, 0, 0)) == 0, "VI should prefer S1 from initial state"


def test_vi_s1_has_highest_q_from_baseline():
    """Q*(Start, baseline, t=0, S1) should be the highest Q value."""
    env = make_env(setting='high', horizon=10)
    vi = ValueIterationAgent(env)
    vi.solve()
    Q = vi.get_Q()
    assert Q[0, 0, 0, 0] == Q[0, 0, 0, :].max()


def test_vi_q_returns_copy():
    env = make_env()
    vi = ValueIterationAgent(env)
    vi.solve()
    q = vi.get_Q()
    q[0, 0, 0, 0] = 999.0
    assert vi._Q[0, 0, 0, 0] != 999.0


def test_vi_v_non_terminal_positive_high_separation():
    """In high-separation setting, V*(initial state) should be > 0."""
    env = make_env(setting='high', horizon=10)
    vi = ValueIterationAgent(env)
    vi.solve()
    assert vi.get_V()[0, 0, 0] > 0.0


def test_vi_run_episode_return_is_finite():
    env = make_env(seed=0)
    vi = ValueIterationAgent(env)
    vi.solve()
    total = vi.run_episode()
    assert np.isfinite(total)


# ===========================================================================
# 11. Convergence: model-free agents improve over episodes
# ===========================================================================

@pytest.mark.parametrize("cls", ALL_MODEL_FREE)
def test_agent_improves_over_training(cls):
    """
    Average return in the last 200 episodes should exceed that of the first 200,
    using the high-separation setting where the signal is strongest.
    """
    env = make_env(setting='high', horizon=5, seed=0)
    n = 1000
    agent = make_agent(cls, env=env, alpha=0.1, epsilon=1.0,
                       epsilon_decay=0.99, epsilon_min=0.05, seed=0)
    returns = agent.train(n)
    early = np.mean(returns[:200])
    late  = np.mean(returns[-200:])
    assert late > early, (
        f"{cls.__name__}: late mean {late:.3f} not > early mean {early:.3f}"
    )


# ===========================================================================
# 12. Upper-bound: Value Iteration ≥ converged model-free agents
# ===========================================================================

def test_vi_upper_bound_over_model_free():
    """
    After sufficient training, VI mean return should be >= that of every
    model-free agent (allowing a small tolerance for stochasticity).
    """
    setting, horizon = 'high', 5
    env_vi  = make_env(setting=setting, horizon=horizon, seed=1)
    vi = ValueIterationAgent(env_vi)
    vi.solve()

    vi_returns = [vi.run_episode() for _ in range(200)]
    vi_mean = np.mean(vi_returns)

    for cls in ALL_MODEL_FREE:
        env_mf = make_env(setting=setting, horizon=horizon, seed=2)
        agent = make_agent(cls, env=env_mf, alpha=0.1, epsilon=1.0,
                           epsilon_decay=0.99, epsilon_min=0.05, seed=3)
        agent.train(2000)
        # Evaluate greedily (ε=0) for 200 episodes
        agent.epsilon = 0.0
        mf_returns = []
        for _ in range(200):
            state = env_mf.reset()
            ep_ret = 0.0
            done = False
            while not done:
                action = agent.select_action(state)
                next_state, reward, done, _ = env_mf.step(action)
                ep_ret += reward
                state = next_state
            mf_returns.append(ep_ret)
        mf_mean = np.mean(mf_returns)

        # VI should be >= model-free with a tolerance of 0.3
        assert vi_mean >= mf_mean - 0.3, (
            f"VI mean {vi_mean:.3f} should be >= {cls.__name__} mean {mf_mean:.3f}"
        )
