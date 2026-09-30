"""Shared plotting style so every chart in the project looks consistent."""

from __future__ import annotations

import matplotlib.pyplot as plt

# Colour-blind-safe pair: blue for treated (emailed), orange for control.
TREATED = "#2a78d6"
CONTROL = "#eb6834"
INK = "#0b0b0b"
MUTED = "#898781"


def apply_style() -> None:
    """Apply a clean, minimal matplotlib style."""
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": MUTED,
            "axes.labelcolor": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#e6e5e1",
            "grid.linewidth": 0.8,
            "xtick.color": INK,
            "ytick.color": INK,
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "figure.dpi": 110,
        }
    )
