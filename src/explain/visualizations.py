import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap #type: ignore
import base64
from io import BytesIO
from pathlib import Path


# ─── shared style helpers ─────────────────────────────────────────────────────

_TITLE_FONT = {"fontsize": 13, "fontweight": "bold", "pad": 10}
_SUBTITLE_FONT = {"fontsize": 9, "color": "#555555"}
_LABEL_FONT = {"fontsize": 10}


def _add_title_block(title: str, subtitle: str = "") -> None:
    """Add a bold title and optional grey subtitle to the *current* plt figure."""
    fig = plt.gcf()
    top = 0.97 if subtitle else 0.96
    fig.text(0.5, top, title, ha="center", va="top",
             fontsize=13, fontweight="bold", color="#1a1a2e")
    if subtitle:
        fig.text(0.5, top - 0.04, subtitle, ha="center", va="top",
                 fontsize=9, color="#555555", style="italic")


def _apply_axis_style(ax, xlabel: str = "", ylabel: str = "") -> None:
    """Apply clean grid + labels to an axis."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if xlabel:
        ax.set_xlabel(xlabel, **_LABEL_FONT)
    if ylabel:
        ax.set_ylabel(ylabel, **_LABEL_FONT)
    ax.tick_params(axis="both", labelsize=9)
    ax.grid(axis="x", linestyle="--", alpha=0.4, color="#cccccc")


# ─── public plot functions ────────────────────────────────────────────────────

def plot_shap_summary(
    shap_values: np.ndarray,
    features: np.ndarray,
    feature_names: list[str],
    out_path: str | Path,
    plot_type: str = "bar",
    max_display: int = 20,
    figsize: tuple[int, int] = (11, 8),
    title: str = "Global Feature Importance (SHAP)",
    subtitle: str = "",
) -> None:
    """Create SHAP summary/bar plot with title and axis labels, saved as PNG."""
    plt.close("all")  # clear any stale state before SHAP creates its own figure

    shap.summary_plot(
        shap_values,
        features,
        feature_names=feature_names,
        plot_type=plot_type,
        max_display=max_display,
        show=False,
    )

    fig = plt.gcf()
    fig.set_size_inches(*figsize)

    # Axis labels
    ax = fig.axes[0]
    if plot_type == "bar":
        ax.set_xlabel("Mean |SHAP Value|  (average impact on model output)", **_LABEL_FONT)
        ax.set_ylabel("Feature", **_LABEL_FONT)
    else:
        ax.set_xlabel("SHAP Value  (impact on model output)", **_LABEL_FONT)
        ax.set_ylabel("Feature", **_LABEL_FONT)
    ax.tick_params(axis="y", labelsize=9)
    ax.tick_params(axis="x", labelsize=9)

    _add_title_block(title, subtitle)

    fig.subplots_adjust(top=0.88, left=0.28, right=0.95, bottom=0.1)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_shap_waterfall(
    shap_values: np.ndarray,
    base_value: float,
    features: np.ndarray,
    feature_names: list[str],
    out_path: str | Path,
    figsize: tuple[int, int] = (10, 7),
    max_display: int = 15,
    title: str = "SHAP Waterfall — Instance Explanation",
    subtitle: str = "",
) -> None:
    """Create SHAP waterfall plot for a single instance with title."""
    plt.close("all")

    explanation = shap.Explanation(
        values=shap_values,
        base_values=base_value,
        data=features,
        feature_names=feature_names,
    )
    shap.plots.waterfall(explanation, max_display=max_display, show=False)

    fig = plt.gcf()
    fig.set_size_inches(*figsize)

    ax = fig.axes[0]
    ax.set_xlabel("SHAP Value  (contribution to model output)", **_LABEL_FONT)
    ax.set_ylabel("Feature  (value)", **_LABEL_FONT)
    ax.tick_params(axis="both", labelsize=9)

    _add_title_block(title, subtitle)

    fig.subplots_adjust(top=0.87, left=0.32, right=0.95, bottom=0.12)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_shap_force(
    shap_values: np.ndarray,
    base_value: float,
    features: np.ndarray,
    feature_names: list[str],
    out_path: str | Path,
    figsize: tuple[int, int] = (14, 4),
    title: str = "SHAP Force Plot — Instance Explanation",
    subtitle: str = "",
) -> None:
    """Create SHAP force plot for a single instance (matplotlib) with title."""
    plt.close("all")

    shap.force_plot(
        base_value,
        shap_values,
        features,
        feature_names=feature_names,
        matplotlib=True,
        show=False,
    )

    fig = plt.gcf()
    fig.set_size_inches(*figsize)

    # Force plot draws its own axis labels; add a suptitle above them
    fig.suptitle(
        f"{title}\n{subtitle}" if subtitle else title,
        fontsize=12,
        fontweight="bold",
        color="#1a1a2e",
        y=1.04,
    )

    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_shap_dependence(
    shap_values: np.ndarray,
    features: np.ndarray,
    feature_names: list[str],
    feature_name: str,
    out_path: str | Path,
    figsize: tuple[int, int] = (7, 5),
    title: str = "",
) -> None:
    """Create SHAP dependence plot for a specific feature."""
    if feature_name not in feature_names:
        raise ValueError(f"Feature '{feature_name}' not found in feature_names")
    feat_idx = feature_names.index(feature_name)

    plt.close("all")

    shap.dependence_plot(
        feat_idx,
        shap_values,
        features,
        feature_names=feature_names,
        show=False,
    )

    fig = plt.gcf()
    fig.set_size_inches(*figsize)

    ax = fig.axes[0]
    ax.set_xlabel(feature_name, **_LABEL_FONT)
    ax.set_ylabel(f"SHAP Value for {feature_name}", **_LABEL_FONT)

    _add_title_block(
        title or f"SHAP Dependence — {feature_name}",
        "How model output changes with feature value",
    )

    fig.subplots_adjust(top=0.87)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ─── base64 helpers ───────────────────────────────────────────────────────────

def encode_plot_to_base64(fig) -> str:
    """Encode matplotlib figure to base64 PNG string."""
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return img_base64


def create_summary_plot_base64(
    shap_values: np.ndarray,
    features: np.ndarray,
    feature_names: list[str],
    plot_type: str = "bar",
    max_display: int = 20,
    figsize: tuple[int, int] = (11, 8),
    title: str = "Global Feature Importance (SHAP)",
    subtitle: str = "",
) -> str:
    """Create summary plot and return as base64 string."""
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
    plot_shap_summary(shap_values, features, feature_names, tmp_path,
                      plot_type, max_display, figsize, title, subtitle)
    with open(tmp_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    os.unlink(tmp_path)
    return b64


def create_waterfall_plot_base64(
    shap_values: np.ndarray,
    base_value: float,
    features: np.ndarray,
    feature_names: list[str],
    max_display: int = 15,
    figsize: tuple[int, int] = (10, 7),
    title: str = "SHAP Waterfall — Instance Explanation",
    subtitle: str = "",
) -> str:
    """Create waterfall plot and return as base64 string."""
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
    plot_shap_waterfall(shap_values, base_value, features, feature_names,
                        tmp_path, figsize, max_display, title, subtitle)
    with open(tmp_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    os.unlink(tmp_path)
    return b64


def create_force_plot_base64(
    shap_values: np.ndarray,
    base_value: float,
    features: np.ndarray,
    feature_names: list[str],
    figsize: tuple[int, int] = (14, 4),
    title: str = "SHAP Force Plot — Instance Explanation",
    subtitle: str = "",
) -> str:
    """Create force plot and return as base64 string."""
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
    plot_shap_force(shap_values, base_value, features, feature_names,
                    tmp_path, figsize, title, subtitle)
    with open(tmp_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    os.unlink(tmp_path)
    return b64