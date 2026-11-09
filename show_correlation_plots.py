"""Visual report for the correlation-analysis part of the lab."""

from pathlib import Path

from correlation_tasks import plot_brain_correlations
from grader_contracts.correlation_tasks import BrainDataInput


plot_brain_correlations(BrainDataInput(str(Path(__file__).parent / "brainsize.txt")))
