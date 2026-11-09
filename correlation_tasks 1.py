"""Задачи второй части лабораторной: корреляционный анализ."""
from __future__ import annotations

import pandas as pd

from grader_contracts.correlation_tasks import BrainCorrelationSummary, BrainDataInput


FEATURES = ["FSIQ", "VIQ", "PIQ", "Weight", "Height"]


def analyze_brain_correlations(data: BrainDataInput) -> BrainCorrelationSummary:
    """Проанализируйте brainsize.txt.

    Разделите наблюдения по полу и для каждой группы вычислите корреляции
    признаков FSIQ, VIQ, PIQ, Weight, Height с MRI_Count методом Пирсона.
    В strongest_mri_feature верните название признака с наибольшим модулем
    корреляции с MRI_Count среди объединённых результатов двух групп.
    """
    frame = pd.read_csv(data.csv_path, sep="\t", na_values="NA")
    women = frame[frame["Gender"] == "Female"]
    men = frame[frame["Gender"] == "Male"]

    def mri_correlations(group: pd.DataFrame) -> dict[str, float]:
        return {
            feature: float(group[feature].corr(group["MRI_Count"]))
            for feature in FEATURES
        }

    women_correlations = mri_correlations(women)
    men_correlations = mri_correlations(men)
    combined = {**women_correlations, **{
        f"men_{feature}": value for feature, value in men_correlations.items()
    }}
    strongest = max(combined, key=lambda feature: abs(combined[feature])).removeprefix("men_")

    return BrainCorrelationSummary(
        men_count=len(men),
        women_count=len(women),
        women_mri_correlation=women_correlations,
        men_mri_correlation=men_correlations,
        strongest_mri_feature=strongest,
    )


def plot_brain_correlations(data: BrainDataInput) -> tuple[object, object]:
    """Show scatter plots and a correlation heatmap for the report."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    frame = pd.read_csv(data.csv_path, sep="\t", na_values="NA")
    scatter = sns.pairplot(
        frame,
        x_vars=FEATURES,
        y_vars=["MRI_Count"],
        hue="Gender",
        height=3,
    )
    heatmap_figure, axis = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        frame[FEATURES + ["MRI_Count"]].corr(),
        annot=True,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        ax=axis,
    )
    axis.set_title("Brain measurements correlation")
    plt.show()
    return scatter.figure, heatmap_figure
