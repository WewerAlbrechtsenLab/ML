from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns


def plot_class_distribution(targets) -> None:
    plt.figure(figsize=(6, 4))
    sns.countplot(x=targets)
    plt.title("Target Distribution")
    plt.tight_layout()


def plot_missingness(features) -> None:
    plt.figure(figsize=(8, 4))
    sns.heatmap(features.isnull(), cbar=False)
    plt.title("Missing Value Matrix")
    plt.tight_layout()
