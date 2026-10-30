"""Run Section 8-1 full evaluation pipeline.

Usage:
    python run_all_evaluation.py
"""

from heuristic_baseline import main as run_baselines
from aggregate_metrics import aggregate
from plot_evaluation import main as run_plots
from hyperparameter_tables import main as run_hparams


def main() -> None:
    run_baselines()
    aggregate()
    run_plots()
    run_hparams()


if __name__ == "__main__":
    main()
