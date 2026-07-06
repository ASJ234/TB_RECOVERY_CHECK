import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import shap #type: ignore
import base64
from io import BytesIO
from pathlib import Path

# Force-plot colours (match SHAP defaults)
_POS_COLOR = "#ff0051"
_NEG_COLOR = "#008bfb"


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
        fig.text(0.5, top - 0.035, subtitle, ha="center", va="top",
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

    fig.subplots_adjust(top=0.88, left=0.30, right=0.95, bottom=0.12)
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
    # Clear SHAP's auto-generated x-axis label so we can replace it
    # without it stacking on top of the E[f(X)] annotation at the bottom.
    ax.set_xlabel("")
    ax.tick_params(axis="both", labelsize=9)
    ax.set_xlabel(
        "SHAP Value  (contribution to model output)",
        fontsize=10,
        labelpad=30,   # push below the E[f(X)] annotation
    )

    _add_title_block(title, subtitle)

    # Extra bottom margin keeps x-label and E[f(X)] annotation separate.
    fig.subplots_adjust(top=0.87, left=0.34, right=0.95, bottom=0.20)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_shap_force(
    shap_values: np.ndarray,
    base_value: float,
    features: np.ndarray,
    feature_names: list[str],
    out_path: str | Path,
    figsize: tuple[int, int] = (14, 5),
    title: str = "SHAP Force Plot — Instance Explanation",
    subtitle: str = "",
) -> None:
    """Custom SHAP-style force plot using grouped horizontal bars.

    Replaces shap.force_plot(matplotlib=True) which renders blank for XGBoost
    tree_path_dependent explainers and causes label overlap for many features.
    This implementation always renders all data and ensures no label overlap.
    """
    plt.close("all")

    sv = np.asarray(shap_values, dtype=float).ravel()
    fnames = list(feature_names)
    fvals = np.asarray(features, dtype=object).ravel()

    if len(sv) != len(fnames):
        raise ValueError(
            f"shap_values length ({len(sv)}) != feature_names length ({len(fnames)})"
        )

    # ── Select top-N features; roll the rest into an 'others' bucket ────────
    MAX_DISPLAY = 12
    order = np.argsort(np.abs(sv))[::-1]
    n_show = min(MAX_DISPLAY, len(sv))
    show_idx = order[:n_show]
    rest_idx = order[n_show:]

    # Sort shown features largest-positive first
    show_idx = sorted(show_idx, key=lambda i: -sv[i])

    other_total = float(np.sum(sv[rest_idx])) if rest_idx.size else 0.0
    shown_sv   = [sv[i] for i in show_idx]
    shown_names = [fnames[i] for i in show_idx]
    shown_vals  = [fvals[i] for i in show_idx]

    # Prepend an 'other features' bucket
    n_others = int(rest_idx.size)
    all_sv    = [other_total] + shown_sv
    all_names = ([f"{n_others} other features"] if n_others else [""]) + shown_names
    all_vals  = [""] + [
        str(v) if not isinstance(v, float) else f"{v:.4g}" for v in shown_vals
    ]

    # ── Build list of bar segments ────────────────────────────────────────────
    segments = []  # (left, width, color, y_label, sv_i)
    cursor = base_value
    for sv_i, name_i, val_i in zip(all_sv, all_names, all_vals):
        if name_i == "":
            cursor += sv_i
            continue
        color = _POS_COLOR if sv_i >= 0 else _NEG_COLOR
        left  = min(cursor, cursor + sv_i)
        width = abs(sv_i)
        y_label = f"{name_i} = {val_i}" if val_i else name_i
        segments.append((left, width, color, y_label, sv_i))
        cursor += sv_i

    f_x = base_value + float(np.sum(sv))
    n_segs = len(segments)

    # ── Figure: auto-height so labels have room ───────────────────────────────
    h = max(figsize[1], 1.8 + 0.50 * n_segs)
    fig, ax = plt.subplots(figsize=(figsize[0], h))

    BAR_H = 0.55
    # Top-to-bottom order on y-axis
    y_positions = list(range(n_segs - 1, -1, -1))

    for y_pos, (left, width, color, y_label, sv_i) in zip(y_positions, segments):
        ax.barh(y_pos, width, left=left, height=BAR_H,
                color=color, alpha=0.85, edgecolor="white", linewidth=0.5)
        # Value text centred inside bar
        sign = "+" if sv_i > 0 else ""
        val_str = f"{sign}{sv_i:.3g}"
        ax.text(left + width / 2, y_pos, val_str,
                ha="center", va="center",
                fontsize=8, fontweight="bold", color="white", zorder=5)

    # ── Y-axis: one tick per feature, no overlap ──────────────────────────────
    ax.set_yticks(y_positions)
    ax.set_yticklabels([seg[3] for seg in segments], fontsize=8.5)
    ax.tick_params(axis="y", pad=4, length=0)

    # Adapt left margin to longest label while keeping enough room for the title block.
    max_label_len = max((len(seg[3]) for seg in segments), default=0)
    left_margin = max(0.30, min(0.60, 0.18 + max_label_len * 0.009))
    fig.subplots_adjust(left=left_margin, right=0.95, top=0.84, bottom=0.16)

    # ── X-axis styling ────────────────────────────────────────────────────────
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel("SHAP Value  (contribution to model output)", fontsize=10)
    ax.tick_params(axis="x", labelsize=9)
    ax.grid(axis="x", linestyle="--", alpha=0.35, color="#cccccc")

    # ── Vertical reference lines ──────────────────────────────────────────────
    ax.axvline(base_value, color="#888888", linewidth=1.0,
               linestyle="--", alpha=0.7, zorder=1)
    ax.axvline(f_x, color="#1a1a2e", linewidth=1.2,
               linestyle="-",  alpha=0.5, zorder=1)

    # ── Annotations above the top bar ─────────────────────────────────────────
    y_ann = n_segs - 0.5
    ax.annotate(
        f"E[f(X)] = {base_value:.4g}",
        xy=(base_value, y_ann), xytext=(base_value, y_ann + 0.6),
        ha="center", va="bottom", fontsize=8.5, color="#888888",
        arrowprops=dict(arrowstyle="-", color="#888888", lw=0.8),
    )
    ax.annotate(
        f"f(x) = {f_x:.4g}",
        xy=(f_x, y_ann), xytext=(f_x, y_ann + 0.6),
        ha="center", va="bottom", fontsize=8.5,
        color="#1a1a2e", fontweight="bold",
        arrowprops=dict(arrowstyle="-", color="#1a1a2e", lw=0.8),
    )

    # ── Legend ────────────────────────────────────────────────────────────────
    pos_patch = mpatches.Patch(color=_POS_COLOR, alpha=0.85,
                               label="Pushes prediction higher")
    neg_patch = mpatches.Patch(color=_NEG_COLOR, alpha=0.85,
                               label="Pushes prediction lower")
    ax.legend(handles=[pos_patch, neg_patch], loc="lower right",
              fontsize=8, framealpha=0.85)

    # ── Title block ───────────────────────────────────────────────────────────
    _add_title_block(title, subtitle)

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
    figsize: tuple[int, int] = (14, 5),
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