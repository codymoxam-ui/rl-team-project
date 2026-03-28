"""
Phase 1 tests: verify project scaffold is complete and well-formed.

Covers:
  - All required directories exist
  - All required files exist
  - requirements.txt contains all expected packages
  - .gitignore excludes large result files and tracks figures
  - All src modules are importable without errors
  - Each module exposes the correct stub class / function names
"""
import importlib
import os

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# 1. Directory structure
# ---------------------------------------------------------------------------

REQUIRED_DIRS = [
    "src",
    "src/env",
    "src/agents",
    "src/experiments",
    "src/analysis",
    "src/visualization",
    "notebooks",
    "results",
    "figures",
    "docs",
    "tests",
]


@pytest.mark.parametrize("dirpath", REQUIRED_DIRS)
def test_required_directory_exists(dirpath):
    full = os.path.join(PROJECT_ROOT, dirpath)
    assert os.path.isdir(full), f"Required directory missing: {dirpath}"


# ---------------------------------------------------------------------------
# 2. Required files
# ---------------------------------------------------------------------------

REQUIRED_FILES = [
    "requirements.txt",
    ".gitignore",
    "README.md",
    "PLAN.md",
    "pytest.ini",
    "src/__init__.py",
    "src/env/__init__.py",
    "src/env/stimulation_env.py",
    "src/agents/__init__.py",
    "src/agents/base_agent.py",
    "src/agents/monte_carlo.py",
    "src/agents/q_learning.py",
    "src/agents/expected_sarsa.py",
    "src/agents/double_q_learning.py",
    "src/agents/value_iteration.py",
    "src/experiments/__init__.py",
    "src/experiments/configs.py",
    "src/experiments/runner.py",
    "src/analysis/__init__.py",
    "src/analysis/metrics.py",
    "src/visualization/__init__.py",
    "src/visualization/plots.py",
    "tests/__init__.py",
]


@pytest.mark.parametrize("filepath", REQUIRED_FILES)
def test_required_file_exists(filepath):
    full = os.path.join(PROJECT_ROOT, filepath)
    assert os.path.isfile(full), f"Required file missing: {filepath}"


# ---------------------------------------------------------------------------
# 3. requirements.txt content
# ---------------------------------------------------------------------------

REQUIRED_PACKAGES = [
    "numpy",
    "matplotlib",
    "seaborn",
    "pandas",
    "jupyter",
    "tqdm",
    "scipy",
    "pytest",
]


@pytest.mark.parametrize("package", REQUIRED_PACKAGES)
def test_requirements_txt_contains_package(package):
    req_path = os.path.join(PROJECT_ROOT, "requirements.txt")
    with open(req_path) as f:
        # Normalise: lower-case, strip extras/version pins for a simple check
        packages_listed = [line.strip().lower().split("[")[0].split("=")[0].split(">")[0].split("<")[0]
                           for line in f if line.strip() and not line.startswith("#")]
    assert package.lower() in packages_listed, (
        f"Package '{package}' missing from requirements.txt"
    )


# ---------------------------------------------------------------------------
# 4. .gitignore content
# ---------------------------------------------------------------------------

def _gitignore_lines():
    gi_path = os.path.join(PROJECT_ROOT, ".gitignore")
    with open(gi_path) as f:
        return [line.strip() for line in f]


def test_gitignore_excludes_results_pkl():
    lines = _gitignore_lines()
    assert any("results" in l and ".pkl" in l for l in lines), (
        ".gitignore should exclude results/*.pkl"
    )


def test_gitignore_excludes_results_npy():
    lines = _gitignore_lines()
    assert any("results" in l and ".npy" in l for l in lines), (
        ".gitignore should exclude results/*.npy"
    )


def test_gitignore_does_not_blanket_exclude_figures():
    lines = _gitignore_lines()
    # "figures/" as a plain ignore rule is not allowed; a negation "!figures/" is fine
    assert "figures/" not in lines, (
        ".gitignore must not unconditionally exclude figures/"
    )


# ---------------------------------------------------------------------------
# 5. Module importability
# ---------------------------------------------------------------------------

IMPORTABLE_MODULES = [
    "src",
    "src.env",
    "src.env.stimulation_env",
    "src.agents",
    "src.agents.base_agent",
    "src.agents.monte_carlo",
    "src.agents.q_learning",
    "src.agents.expected_sarsa",
    "src.agents.double_q_learning",
    "src.agents.value_iteration",
    "src.experiments",
    "src.experiments.configs",
    "src.experiments.runner",
    "src.analysis",
    "src.analysis.metrics",
    "src.visualization",
    "src.visualization.plots",
]


@pytest.mark.parametrize("module", IMPORTABLE_MODULES)
def test_module_importable(module):
    try:
        importlib.import_module(module)
    except ImportError as exc:
        pytest.fail(f"Cannot import '{module}': {exc}")


# ---------------------------------------------------------------------------
# 6. Stub class / function names
# ---------------------------------------------------------------------------

def test_stub_BaseAgent():
    from src.agents.base_agent import BaseAgent
    assert BaseAgent is not None


def test_stub_MonteCarloAgent():
    from src.agents.monte_carlo import MonteCarloAgent
    assert MonteCarloAgent is not None


def test_stub_QLearningAgent():
    from src.agents.q_learning import QLearningAgent
    assert QLearningAgent is not None


def test_stub_ExpectedSARSAAgent():
    from src.agents.expected_sarsa import ExpectedSARSAAgent
    assert ExpectedSARSAAgent is not None


def test_stub_DoubleQLearningAgent():
    from src.agents.double_q_learning import DoubleQLearningAgent
    assert DoubleQLearningAgent is not None


def test_stub_ValueIterationAgent():
    from src.agents.value_iteration import ValueIterationAgent
    assert ValueIterationAgent is not None


def test_stub_StimulationEnv():
    from src.env.stimulation_env import StimulationEnv
    assert StimulationEnv is not None


def test_stub_ExperimentConfig():
    from src.experiments.configs import ExperimentConfig
    assert ExperimentConfig is not None


def test_stub_get_all_configs():
    from src.experiments.configs import get_all_configs
    assert callable(get_all_configs)


def test_stub_run_experiment():
    from src.experiments.runner import run_experiment
    assert callable(run_experiment)


def test_stub_run_all():
    from src.experiments.runner import run_all
    assert callable(run_all)


def test_stub_smooth():
    from src.analysis.metrics import smooth
    assert callable(smooth)


def test_stub_compute_summary():
    from src.analysis.metrics import compute_summary
    assert callable(compute_summary)


def test_stub_plot_learning_curves():
    from src.visualization.plots import plot_learning_curves
    assert callable(plot_learning_curves)


def test_stub_plot_performance_heatmap():
    from src.visualization.plots import plot_performance_heatmap
    assert callable(plot_performance_heatmap)
