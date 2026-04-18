# Adaptive Multi-Site Stimulation Control Using Reinforcement Learning in an MDP

**Learning Sequential Stimulation Policies Under State-dependent EEG Response Dynamics**

**Authors:** Fatih Karatay and Cody Moxam
**Course:** EN.705.741.8VL.SP26
**Date:** April 2026

---

## Abstract

Adaptive neurostimulation is not always a static target-selection problem because stimulation history may alter later responsiveness and reward quality. We model adaptive multi-site stimulation as a finite-horizon Markov decision process in which an agent selects among four stimulation sites while patient-response state evolves among baseline, receptive, and non-receptive modes. Four model-free reinforcement-learning methods -- Monte Carlo Control, Q-Learning, Expected SARSA, and Double Q-Learning -- were compared with a Value Iteration benchmark across response-separation levels, episode horizons, and switching costs (90 configurations; 10 seeds each). Temporal-difference methods consistently outperformed Monte Carlo Control (Welch's t-test, p < 0.05 in all settings), while no statistically significant differences were detected among Q-Learning, Expected SARSA, and Double Q-Learning (all pairwise p > 0.17). Value Iteration achieved higher return in the high- and moderate-separation settings (p < 0.05). Increasing switching cost reduced switching frequency (Spearman $\rho$ = -0.83 to -0.94, p < 0.001) but increased non-receptive state occupancy ($\rho$ = 0.76 to 0.85, p < 0.001), revealing a control tradeoff. Contrary to expectation, Double Q-Learning did not exhibit significantly lower return variance than Q-Learning (Levene's test, p > 0.65 in all settings). These results support framing state-dependent adaptive stimulation as a sequential control problem rather than static arm selection.

## 1. Introduction

Brain stimulation site selection is often framed as choosing the single best target. In a static formulation, the problem resembles a multi-armed bandit: reward is sampled from fixed action distributions and the optimal strategy is to identify the highest-value arm. Clinical neurostimulation, however, is often history-dependent rather than static. Repeated stimulation can alter subsequent tissue responsiveness, so current actions may influence future reward quality and future state occupancy. When that occurs, site selection is more naturally posed as a sequential control problem than as one-step arm selection, consistent with prior adaptive neurostimulation formulations (Pineau et al., 2009).

The present study models adaptive multi-site stimulation as a finite-horizon Markov decision process (MDP). At each timestep an agent chooses among four stimulation sites. The observed EEG response and resulting reward depend on both the selected site and the patient-response state, which can be baseline, receptive, or non-receptive. Repeated stimulation of the same site increases the probability of occupying the non-receptive state, whereas switching promotes recovery toward baseline. An optional switching cost penalizes excessive site changes and creates an explicit tradeoff between immediate penalty avoidance and longer-horizon state management. This formulation is clinically motivated but intentionally simplified, and is used here to isolate the control consequences of state-dependent responsiveness rather than to represent a clinically calibrated physiological model.

Four model-free reinforcement-learning methods are evaluated against a model-based Value Iteration benchmark: Monte Carlo Control, Q-Learning, Expected SARSA, and Double Q-Learning. The study varies response separation, episode horizon, and switching cost in order to test how environmental informativeness and control penalties shape learned policies. We evaluate the following hypotheses:

**H1.** Agents trained in higher-separation settings achieve higher mean final return than agents trained in lower-separation settings.

**H2.** Higher switching cost is associated with lower switching frequency and higher non-receptive state occupancy in converged policies.

**H3.** Double Q-Learning exhibits lower variance in per-seed mean final return compared to Q-Learning.

**H4.** Model-free agents achieve lower mean final return than the Value Iteration benchmark.

**H5.** Converged policies develop structured site preferences aligned with the reward model, rather than selecting sites uniformly.

## 2. Methods

### 2.1 MDP Formulation

The environment is a finite-horizon, fully observed, episodic MDP defined by the following components:

**State.** $s_t = (\text{current\_site}, \text{patient\_state}, t)$, where current\_site $\in \{\text{Start}, S_1, S_2, S_3, S_4\}$, patient\_state $\in \{\text{baseline}, \text{receptive}, \text{non-receptive}\}$, and $t \in \{0, 1, \dots, H\}$ is the timestep within the episode. The initial state is $s_0 = (\text{Start}, \text{baseline}, 0)$.

**Actions.** $a_t \in \{S_1, S_2, S_3, S_4\}$ -- the agent selects one of four stimulation sites at each step.

**Patient-state transitions.** After the agent selects site $a_t$, the patient state transitions stochastically according to $T(p_{t+1} \mid p_t, \text{same\_site})$, where same\_site indicates whether the chosen site matches the current site. Table 1 specifies the transition probabilities. Repeated stimulation of the same site increases the probability of moving toward non-receptive, while switching promotes recovery toward baseline.

**Table 1. Patient-state transition probabilities $T(p' \mid p, \text{same\_site})$.**

| Current State | Action Type | $\rightarrow$ baseline | $\rightarrow$ receptive | $\rightarrow$ non-receptive |
|---|---|---|---|---|
| baseline | any | 0.60 | 0.30 | 0.10 |
| receptive | same site | 0.20 | 0.30 | 0.50 |
| receptive | different site | 0.50 | 0.30 | 0.20 |
| non-receptive | same site | 0.10 | 0.10 | 0.80 |
| non-receptive | different site | 0.40 | 0.30 | 0.30 |

**Observation and reward.** Each stimulation produces an EEG observation $o_{t+1} \sim P(o \mid a_t, p_t)$ drawn from a site- and state-dependent distribution. The observation maps deterministically to reward: $g(\text{favorable}) = +1$, $g(\text{neutral}) = 0$, $g(\text{unfavorable}) = -1$. An optional switching penalty applies when the agent changes site:

$$r_t = g(o_{t+1}) - c_{\text{switch}} \cdot \mathbf{1}[a_t \neq \text{current\_site}]$$

Table 2 gives the observation probabilities for the high-separation setting, in which sites are clearly differentiated and state-dependent reward degradation is most pronounced. The moderate-separation setting compresses between-site differences (e.g., S1 baseline favorable probability drops from 0.70 to 0.60) while preserving the same patient-state ordering. In the low-separation setting, all sites have nearly identical observation distributions, so policy differences must arise primarily from patient-state management.

**Table 2. EEG observation probabilities $P(o \mid \text{site}, \text{patient\_state})$ -- high-separation setting.**

| Site | Patient State | Favorable | Neutral | Unfavorable | E[reward] |
|---|---|---|---|---|---|
| S1 | baseline | 0.70 | 0.20 | 0.10 | +0.60 |
| S1 | receptive | 0.85 | 0.10 | 0.05 | +0.80 |
| S1 | non-receptive | 0.30 | 0.40 | 0.30 | 0.00 |
| S2 | baseline | 0.50 | 0.30 | 0.20 | +0.30 |
| S2 | receptive | 0.65 | 0.25 | 0.10 | +0.55 |
| S2 | non-receptive | 0.20 | 0.40 | 0.40 | -0.20 |
| S3 | baseline | 0.30 | 0.40 | 0.30 | 0.00 |
| S3 | receptive | 0.45 | 0.35 | 0.20 | +0.25 |
| S3 | non-receptive | 0.15 | 0.35 | 0.50 | -0.35 |
| S4 | baseline | 0.45 | 0.35 | 0.20 | +0.25 |
| S4 | receptive | 0.60 | 0.30 | 0.10 | +0.50 |
| S4 | non-receptive | 0.25 | 0.40 | 0.35 | -0.10 |

**Objective.** The agent maximizes expected undiscounted episodic return:

$$\max_{\pi} \; \mathbb{E}_{\pi}\!\left[\sum_{t=0}^{H-1} r_t\right]$$

**State space size.** With 5 site values, 3 patient states, and $H+1$ timesteps, the state space has $|S| = 5 \times 3 \times (H+1)$ states. For $H = 10$, $|S| = 165$ and $|S| \times |A| = 660$, making the problem fully tractable with tabular methods.

### 2.2 Algorithms

All agents use a tabular Q-table indexed by $(s, a)$ with $\varepsilon$-greedy exploration. Exploration decays exponentially: $\varepsilon_k = \max(\varepsilon_{\min},\; \varepsilon_0 \cdot d^k)$ where $k$ is the episode number, $\varepsilon_0 = 1.0$, $d = 0.995$, and $\varepsilon_{\min} = 0.05$. The learning rate is $\alpha = 0.1$ and the discount factor is $\gamma = 1.0$ (appropriate for finite-horizon tasks where total return matters).

Four model-free methods are compared. Monte Carlo Control (Sutton & Barto, 2018, Ch. 5) is on-policy and first-visit, updating Q-values from complete episode returns. Q-Learning (Watkins & Dayan, 1992) uses off-policy TD(0) with a greedy bootstrap target. Expected SARSA (van Seijen et al., 2009) uses the expected value under the current $\varepsilon$-greedy policy as its bootstrap target. Double Q-Learning (Hasselt, 2010) maintains two Q-tables to reduce maximization bias, with action selection based on $(Q_1 + Q_2)/2$. Value Iteration (Sutton & Barto, 2018, Ch. 4) computes the optimal finite-horizon policy by backward induction using the known transition and observation models and is evaluated over 200 stochastic rollout episodes per seed.

### 2.3 Experimental Design

Experiments follow a full-factorial design: 3 response-separation settings (high, moderate, low) $\times$ 2 episode horizons ($H \in \{5, 10\}$) $\times$ 3 switching costs ($c_{\text{switch}} \in \{0, 0.1, 0.25\}$) $\times$ 5 algorithms = 90 configurations. Each model-free configuration is run across 10 independent random seeds with 5,000 training episodes per seed. Value Iteration computes the exact optimal policy and evaluates it over 200 stochastic rollout episodes per seed.

The primary analysis focuses on the representative long-horizon, no-switching-cost setting ($H = 10$, $c_{\text{switch}} = 0$). Switching-cost effects are analyzed across all three cost levels in the high-separation setting.

### 2.4 Performance Evaluation and Statistical Methods

**Performance measure.** The primary metric is the *mean final return*: the average episodic return computed over the last 10% of training episodes (episodes 4,501--5,000), then averaged across the 10 seeds. At this point in training, $\varepsilon$ has decayed to its floor of 0.05, so the agent follows a near-greedy policy. Each seed produces one scalar (its mean return over episodes 4,501--5,000), yielding 10 independent observations per algorithm-setting combination.

**Pairwise algorithm comparisons.** To compare two algorithms within the same setting, we apply Welch's t-test (unequal-variance two-sample t-test) to the 10 per-seed mean final returns of each algorithm. We report p-values and consider results significant at the $\alpha = 0.05$ level. This test is appropriate because the per-seed means are independent observations and the Central Limit Theorem supports approximate normality of the sample means.

**Variance comparisons (H3).** To test whether Double Q-Learning produces lower return variance than Q-Learning, we apply Levene's test to the per-seed mean final returns.

**Association tests (H2).** To test the relationship between switching cost and behavioral metrics, we compute Spearman rank correlations across the three cost levels using per-seed measurements (30 observations per algorithm: 10 seeds $\times$ 3 cost levels).

## 3. Results

### 3.1 Reward separability improves learning performance (H1)

Figure 1 shows learning curves for all five methods at $H = 10$, $c_{\text{switch}} = 0$. In the high-separation setting, the temporal-difference methods stabilize near mean episode returns of 4.6--4.7, whereas Monte Carlo stabilizes lower near 4.1. In moderate separation the same ranking persists but shifts downward to roughly 3.1--3.3 for the temporal-difference methods. In low separation, all model-free methods flatten near 1.8--2.0. Value Iteration provides the upper bound in all settings.

Welch's t-tests confirm that H1 is strongly supported: every model-free algorithm achieves significantly higher mean final return in the high-separation setting than in the low-separation setting (all p < 0.001). For example, Q-Learning achieves 4.589 $\pm$ 0.144 in high separation versus 1.951 $\pm$ 0.113 in low separation ($t = 43.3$, $p < 0.001$).

![Figure 1](../figures/fig1_learning_curves_h10_c0.0.png)

*Figure 1. Learning curves for $H = 10$, $c_{\text{switch}} = 0$. Smoothed mean $\pm$ 1 std across 10 seeds. Value Iteration (dashed) provides the upper bound. Temporal-difference methods stabilize above Monte Carlo in all settings, with separation level controlling the achievable return scale.*

Table 3 reports mean final return $\pm$ standard deviation for all algorithm-setting combinations at $H = 10$, $c_{\text{switch}} = 0$. Monte Carlo is significantly below all three temporal-difference methods in every setting (Welch's t-test, p < 0.05; e.g., MC vs. Q-Learning in high separation: $t = -5.58$, $p < 0.001$). No statistically significant pairwise differences are detected among Q-Learning, Expected SARSA, and Double Q-Learning (all pairwise p > 0.17).

**Table 3. Mean final return $\pm$ std ($H = 10$, $c_{\text{switch}} = 0$, $n = 10$ seeds).**

| Algorithm | High Separation | Moderate Separation | Low Separation |
|---|---|---|---|
| Value Iteration | 5.235 $\pm$ 0.436 | 3.880 $\pm$ 0.589 | 2.340 $\pm$ 0.530 |
| Expected SARSA | 4.674 $\pm$ 0.107 | 3.323 $\pm$ 0.090 | 1.965 $\pm$ 0.139 |
| Double Q-Learning | 4.593 $\pm$ 0.147 | 3.317 $\pm$ 0.141 | 1.942 $\pm$ 0.137 |
| Q-Learning | 4.589 $\pm$ 0.144 | 3.260 $\pm$ 0.144 | 1.951 $\pm$ 0.113 |
| Monte Carlo | 4.081 $\pm$ 0.232 | 3.059 $\pm$ 0.140 | 1.801 $\pm$ 0.128 |

### 3.2 Gap to optimal benchmark (H4)

Figure 2 shows the gap between each model-free algorithm's mean final return and the Value Iteration benchmark. In the high-separation setting, all model-free methods are significantly below Value Iteration (Welch's t-test, p < 0.01). In moderate separation, the gap remains significant for all methods (p < 0.05). In low separation, Monte Carlo remains significantly below Value Iteration ($p = 0.014$), whereas the temporal-difference methods are only marginally non-significant ($p = 0.054$--$0.066$), reflecting both the smaller absolute gap and the higher variance of Value Iteration rollouts in this setting. The temporal-difference methods recover 84--89% of the Value Iteration return across settings.

![Figure 2](../figures/fig7_algorithm_gap_h10_c0.0.png)

*Figure 2. Gap to optimal (VI return $-$ agent return) at $H = 10$, $c_{\text{switch}} = 0$. Monte Carlo has the largest gap in all settings. The three temporal-difference methods cluster together with smaller gaps.*

### 3.3 Switching cost shapes policy behavior (H2)

Figure 3 and Table 4 summarize the switching-cost sweep in the high-separation setting. For every algorithm, switching frequency decreases monotonically as $c_{\text{switch}}$ increases from 0 to 0.25, while non-receptive state occupancy rises in parallel.

Spearman rank correlations confirm strong statistical associations. For each model-free algorithm, switching frequency is negatively correlated with $c_{\text{switch}}$ ($\rho = -0.83$ to $-0.94$, all $p < 0.001$) and non-receptive occupancy is positively correlated with $c_{\text{switch}}$ ($\rho = 0.76$ to $0.85$, all $p < 0.001$). These results support H2.

**Table 4. Switching rate and non-receptive fraction by switching cost (high separation, $H = 10$).**

| Algorithm | $c = 0$ switch rate | $c = 0.1$ | $c = 0.25$ | $c = 0$ non-rec | $c = 0.1$ | $c = 0.25$ |
|---|---|---|---|---|---|---|
| Monte Carlo | 0.625 | 0.551 | 0.427 | 0.195 | 0.200 | 0.237 |
| Q-Learning | 0.511 | 0.401 | 0.300 | 0.205 | 0.230 | 0.273 |
| Expected SARSA | 0.464 | 0.376 | 0.275 | 0.214 | 0.233 | 0.275 |
| Double Q-Learning | 0.467 | 0.398 | 0.284 | 0.218 | 0.242 | 0.278 |

![Figure 3](../figures/fig5_switching_vs_cost_high_h10.png)

*Figure 3. Switching-cost effects in the high-separation setting, $H = 10$. Left: switch rate decreases with cost. Right: non-receptive state occupancy increases with cost. All trends are statistically significant (Spearman $|\rho| > 0.76$, $p < 0.001$).*

### 3.4 Site preference and state-aware behavior (H5)

Figure 4 shows converged site-visit frequencies in the high-separation setting. All algorithms concentrate the majority of their actions on S1, the site with the highest expected reward in the baseline and receptive states (Table 2). Value Iteration allocates approximately 77% of actions to S1 and nearly zero to S3. The temporal-difference methods allocate 67--71% to S1. Monte Carlo is more diffuse, with only 53% on S1 and larger shares on S2 and S4. All algorithms maintain baseline-plus-receptive occupancy of 79--84% of episode time, with Value Iteration achieving the lowest non-receptive occupancy (approximately 0.175 across settings). These non-uniform action distributions support H5.

![Figure 4](../figures/fig4_site_freq_high_h10_c0.0.png)

*Figure 4. Converged site-visit frequency in the high-separation setting, $H = 10$, $c_{\text{switch}} = 0$. All methods prefer S1. Value Iteration concentrates most aggressively; Monte Carlo distributes visits most broadly.*

### 3.5 Stability comparison: Q-Learning vs. Double Q-Learning (H3)

Levene's test for equality of variances finds no significant difference between Q-Learning and Double Q-Learning in any setting: $p = 0.780$ (high), $p = 0.920$ (moderate), $p = 0.650$ (low). The observed variances are nearly identical (e.g., 0.0206 vs. 0.0216 in high separation). H3 is **not supported**: Double Q-Learning does not exhibit meaningfully lower return variance than Q-Learning in this experimental suite.

### 3.6 Summary of hypothesis outcomes

**Table 5. Summary of hypothesis outcomes.**

| Hypothesis | Outcome | Test | Key Result |
|---|---|---|---|
| H1: Higher separation $\rightarrow$ higher return | Supported | Welch's t-test | High vs. low: all algorithms $p < 0.001$ |
| H2: Higher $c_{\text{switch}}$ $\rightarrow$ less switching, more non-receptive | Supported | Spearman correlation | $|\rho| > 0.76$, $p < 0.001$ for all algorithms |
| H3: Double Q-Learning has lower return variance | Not supported | Levene's test | $p > 0.65$ in all settings |
| H4: Model-free $<$ Value Iteration | Supported | Welch's t-test | Significant in high and moderate settings ($p < 0.05$) |
| H5: Converged policies show structured site preferences | Supported | Qualitative | S1 concentration 53--77% vs. uniform 25% baseline |

## 4. Discussion

The results support the central modeling claim of this study: once patient responsiveness depends on stimulation history, site selection becomes a sequential decision-making problem rather than a static bandit. The switching-cost analysis (H2) makes this especially clear. In a bandit setting, switching cost would simply discourage alternation and the optimal strategy would consist of finding the best arm and staying with it. In this MDP, however, staying at one site for too long increases the probability of transitioning into non-receptive states, which degrades future reward quality. The coupled decrease in switching frequency and increase in non-receptive occupancy observed with higher $c_{\text{switch}}$ (Table 4) exposes a control dilemma that has no counterpart in a bandit formulation: effective policies must balance the immediate switching penalty against delayed state degradation.

The most notable algorithmic finding is the statistical equivalence among the three temporal-difference methods across all separation settings. Q-Learning (off-policy, max-based bootstrap), Expected SARSA (on-policy, expected-value bootstrap), and Double Q-Learning (bias-reduced, dual-table bootstrap) differ substantively in their update mechanics, yet no pairwise difference was detected with the present sample size in this tabular environment. This suggests that in a small state-action space ($|S \times A| = 660$ for $H = 10$), the choice of bootstrap target matters less than the use of bootstrapping itself. The consistently lower performance of Monte Carlo Control is compatible with this interpretation: single-step bootstrapping propagates reward information through patient-state transitions faster than the end-of-episode returns used by Monte Carlo, which is particularly consequential in a finite-horizon MDP where present actions influence later reward through state dynamics.

The failure to reject H3 deserves specific attention. Double Q-Learning is designed to reduce maximization bias associated with the max operator in Q-Learning (Hasselt, 2010). In this environment, the small state-action space and bounded rewards ($r \in [-1, +1]$ before accounting for switching cost) leave limited scope for maximization bias to cause instability. The absence of a variance advantage should not be interpreted as a general algorithmic equivalence; in larger or more complex environments, the bias-reduction mechanism of Double Q-Learning may become more consequential.

The absence of statistically significant pairwise differences among the three temporal-difference methods should not be interpreted as formal algorithmic equivalence. The correct inference is narrower: with the present sample size and this small tabular environment, no pairwise difference was detected. Larger state-action spaces, function approximation, or longer horizons could reveal performance separations not visible here.

## 5. Limitations

This environment is intended to be a simplified simulation that generates interpretable RL structure, as opposed to a biophysically validated clinical model. The transition dynamics and EEG response distributions (Tables 1--2) have been selected to generate interesting tradeoffs, without clinical calibration.

The state space and observation spaces are intentionally discrete and small, which allows for a fair comparison between different tabular RL techniques, but renders the task less immediately applicable to the clinical setting. A natural next step could involve making the problem partially observable, with the patient's true state not being directly observed (a clinically relevant extension to the POMDP formulation).

No explicit static-policy or contextual-bandit baseline was evaluated. Claims about superiority over a static controller are supported by the structured, state-dependent nature of the learned policies (H5) rather than by a direct head-to-head comparison.

The model-free agents are evaluated at the exploration floor $\varepsilon = 0.05$, whereas Value Iteration is evaluated as a deterministic optimal policy. This slightly favors the benchmark, though the resulting gap is bounded above by $\varepsilon \times$ (random-greedy return difference).

The primary analysis examines a representative slice of the parameter space ($H = 10$, $c_{\text{switch}} = 0$) and a switching-cost sweep. All 90 results from the complete configuration sweep have been independently computed. The same qualitative algorithm ranking holds for $H = 5$, with lower achievable returns due to the shorter episode.

## 6. Conclusion

Adaptive multi-site stimulation was formulated as a finite-horizon MDP with state-dependent EEG-derived reward dynamics. Across 90 configurations, the three temporal-difference methods consistently outperformed Monte Carlo Control, Value Iteration remained the upper-bound benchmark, and increasing switching cost produced a clear tradeoff between switch frequency and non-receptive state occupancy. No statistically significant variance advantage for Double Q-Learning over Q-Learning was detected in this small tabular setting. Overall, the results support reinforcement learning as an appropriate framework for history-dependent stimulation control and motivate future extensions to partially observable and clinically calibrated environments.

## References

- Hasselt, H. van. (2010). Double Q-Learning. *Advances in Neural Information Processing Systems*, 23, 2613--2621.
- Pineau, J., Guez, A., Vincent, R., Panuccio, G., & Bhomick, S. (2009). Treating epilepsy via adaptive neurostimulation: A reinforcement learning approach. *International Journal of Neural Systems*, 19(4), 227--240.
- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
- van Seijen, H., van Hasselt, H., Whiteson, S., & Wiering, M. (2009). A theoretical and empirical analysis of Expected Sarsa. *IEEE Symposium on Adaptive Dynamic Programming and Reinforcement Learning*, 177--184.
- Watkins, C. J. C. H., & Dayan, P. (1992). Q-Learning. *Machine Learning*, 8(3--4), 279--292.
