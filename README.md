# Adaptive Multi-Site Stimulation Control Using Reinforcement Learning in an MDP

Code and analysis for the manuscript:

**Karatay F, Moxam C. _Adaptive Multi-Site Stimulation Control Using Reinforcement Learning in an MDP_.**

This repository studies adaptive multi-site stimulation as a **sequential control problem** rather than a static target-selection problem. We model stimulation-site selection as a finite-horizon Markov decision process (MDP) in which stimulation history influences a latent **stimulation-response state**, which in turn affects EEG-derived reward quality. The project compares several tabular reinforcement-learning methods against a model-based benchmark across multiple environment settings.

## Overview

In many simplified formulations, stimulation-site selection can be treated like a multi-armed bandit: each site has a fixed reward distribution, and the goal is to identify the best arm. This repository instead examines a history-dependent setting in which repeated stimulation can alter later responsiveness. Under those conditions, current actions affect future state occupancy and future reward quality, making the problem inherently sequential.

The code implements:

- A finite-horizon MDP for adaptive multi-site stimulation
- A state-dependent EEG observation and reward model
- Four model-free reinforcement-learning methods:
  - Monte Carlo Control
  - Q-Learning
  - Expected SARSA
  - Double Q-Learning
- A model-based Value Iteration benchmark
- Full-factorial experiments varying:
  - response separation
  - episode horizon
  - switching cost
- Statistical analyses and figure generation used in the manuscript

## Main question

The central question is whether adaptive stimulation-site selection should be treated as:

- a **static target-selection problem**, or
- a **sequential control problem** in which stimulation history changes later reward structure.

The simulations in this repository support the second view.

## Repository structure

The exact structure may evolve slightly, but the repository is organized around the following components:

```text
.
├── src/                  # Core environment, agents, experiments, analysis, visualization
│   ├── agents/           # RL algorithms
│   ├── env/              # MDP environment and dynamics
│   ├── experiments/      # Experiment runners
│   ├── analysis/         # Statistical analysis utilities
│   └── visualization/    # Plotting and figure generation
├── scripts/              # End-to-end pipeline scripts
├── results/              # Generated experiment outputs
├── figures/              # Generated figures
└── README.md
