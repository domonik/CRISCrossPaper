import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from scipy.stats import wilcoxon
from plotly_template import COLORS, WIDTH, PT8, PT6, PT7
from statsmodels.stats.multitest import multipletests
from plotly.subplots import make_subplots
import numpy as np

NAME_FIX = {
    "EX": "Ex",
    "CRISPRofft": "CRISPR-OFFT",
    "cnnCRISPR": "CnnCrispr",
}


# ---------------------------------------------------------------------------
# Significance label helpers
# ---------------------------------------------------------------------------

def stars(p):
    """Returns star string or empty string (no n.s.)."""
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    return ""


def stars_label(p):
    """Returns star string or 'n.s.' – always produces a label."""
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    return "n.s."


# ---------------------------------------------------------------------------
# Legacy bracket helper (kept for external callers)
# ---------------------------------------------------------------------------

def add_stars_with_stripes(
    tcell_results, strategy_order, tcell_df, fig, subplot_idx,
    strategy_col_prefix="Strategy", space_fac=0.15, show_ns=True,
):
    strategy_to_x = {s: i for i, s in enumerate(strategy_order)}
    y_max = tcell_df["mean"].max() + tcell_df["std"].max()
    line_spacing = space_fac * 0.15
    current_y = y_max + line_spacing

    for idx, row in tcell_results.iterrows():
        s1 = row[f"{strategy_col_prefix} 1"]
        s2 = row[f"{strategy_col_prefix} 2"]
        pval = row["padj"]

        if np.isnan(pval):
            continue
        elif pval < 0.001:
            star_label = "***"
        elif pval < 0.01:
            star_label = "**"
        elif pval < 0.05:
            star_label = "*"
        else:
            if show_ns:
                star_label = "n.s."
            else:
                continue

        x0 = strategy_to_x[s1]
        x1 = strategy_to_x[s2]

        fig.add_shape(
            type="line", x0=x0, x1=x1, y0=current_y, y1=current_y,
            xref=f"x{subplot_idx}", yref=f"y{subplot_idx}",
            line=dict(color="black", width=1),
        )
        fig.add_annotation(
            x=(x0 + x1) / 2,
            y=current_y if star_label == "n.s." else current_y - 0.05,
            yanchor="bottom", text=star_label, showarrow=False,
            xref=f"x{subplot_idx}", yref=f"y{subplot_idx}",
            font=dict(size=12) if "*" in star_label else dict(size=PT6),
        )
        current_y += line_spacing


# ---------------------------------------------------------------------------
# Significance bracket drawing
# ---------------------------------------------------------------------------

def add_significance_brackets(
    fig,
    comparisons,
    y_start,
    y_step,
    row=1,
    col=1,
    ncols=3,
    tick_frac=0.35,
    font_size_stars=12,
):
    """
    Draw stacked horizontal significance brackets above grouped bars.

    Parameters
    ----------
    comparisons : list of (x0, x1, label)
        x0 / x1 are *numeric* categorical positions (0-based index in the
        category order).  label is "***", "**", "*", or "n.s.".
    y_start : float
        Data-coordinate y level for the lowest bracket.
    y_step : float
        Vertical spacing between brackets in data coordinates.
    ncols : int
        Total number of subplot columns in this figure (used to derive
        axis references).
    tick_frac : float
        Fraction of y_step for the downward tick marks at bracket ends.
    """
    # Axis references for a single-row figure
    idx = (row - 1) * ncols + col
    xref = "x" if idx == 1 else f"x{idx}"
    yref = "y" if idx == 1 else f"y{idx}"
    tick_h = y_step * tick_frac

    # Sort by span width: shorter comparisons sit lower to minimise crossings
    sorted_comps = sorted(comparisons, key=lambda c: abs(c[1] - c[0]))

    for i, (x0, x1, label) in enumerate(sorted_comps):
        y = y_start + i * y_step

        # Horizontal bar
        fig.add_shape(
            type="line", x0=x0, x1=x1, y0=y, y1=y,
            xref=xref, yref=yref,
            line=dict(color="black", width=1),
        )
        # Left downward tick
        fig.add_shape(
            type="line", x0=x0, x1=x0, y0=y - tick_h, y1=y,
            xref=xref, yref=yref,
            line=dict(color="black", width=1),
        )
        # Right downward tick
        fig.add_shape(
            type="line", x0=x1, x1=x1, y0=y - tick_h, y1=y,
            xref=xref, yref=yref,
            line=dict(color="black", width=1),
        )
        # Label centred above the horizontal bar
        is_stars = "*" in label
        fig.add_annotation(
            x=(x0 + x1) / 2, y=y,
            yanchor="bottom", xanchor="center",
            text=label, showarrow=False,
            xref=xref, yref=yref,
            font=dict(size=font_size_stars if is_stars else PT6),
        )


# ---------------------------------------------------------------------------
# Bar-value annotations
# ---------------------------------------------------------------------------

def add_bar_values(
    fig, pdf, modes, row=1, col=1,
    cat_col="window_size", x_val="mode",
    outside=True, font_color="black", offset_x=0.02,
):
    """Annotate numeric mean values on top of each bar."""
    if cat_col in pdf.columns:
        windows = list(pdf[cat_col].unique())
        use_windows = True
    else:
        windows = [1]
        use_windows = False

    n_groups = len(windows)
    group_width = 0.7
    box_width = group_width / n_groups

    for i, window in enumerate(windows):
        offset = -group_width / 2 + box_width / 2 + i * box_width - offset_x

        for model in modes:
            if use_windows:
                bar = pdf[(pdf[x_val] == model) & (pdf[cat_col] == window)]
            else:
                bar = pdf[pdf[x_val] == model]
            if bar.empty:
                continue

            mean_val = bar["mean"].values[0]
            x_numeric = modes.index(model) + offset
            fig.add_annotation(
                x=x_numeric, y=mean_val,
                text=f"{mean_val:.3f}", showarrow=False,
                xref="x", yref="y", row=row, col=col,
                yanchor="bottom" if outside else "top",
                xanchor="center", textangle=90,
                font=dict(color=font_color),
            )


# ---------------------------------------------------------------------------
# Statistical testing
# ---------------------------------------------------------------------------

def calc_sig_fig1(pdf_23_raw, reference_mode="Base"):
    """
    Figure 1, panel b: Wilcoxon test of every non-reference mode at
    window_size == 23 against the reference mode at the same window size.

    Parameters
    ----------
    pdf_23_raw : DataFrame
        Split-level AUCPR data filtered to window_size == 23.
        Must contain columns: mode, split, AUCPR.
    reference_mode : str
        The mode to test against (default "Base").

    Returns
    -------
    DataFrame with columns: mode, p_value, padj
    """
    ref_data = pdf_23_raw[pdf_23_raw["mode"] == reference_mode].copy()
    results = []

    for mode, mode_data in pdf_23_raw.groupby("mode"):
        if mode == reference_mode:
            continue
        combined = pd.concat([ref_data, mode_data])
        pivot = combined.pivot_table(index="split", columns="mode", values="AUCPR")
        if mode in pivot.columns and reference_mode in pivot.columns:
            stat, p = wilcoxon(pivot[mode], pivot[reference_mode])
            results.append({"mode": mode, "p_value": p})

    if not results:
        return pd.DataFrame(columns=["mode", "p_value", "padj"])
    df = pd.DataFrame(results)
    _, df["padj"], _, _ = multipletests(df["p_value"])
    return df


def calc_sig_ablation(abl_df_raw, reference_mode="AG"):
    """
    Figure 1, panel c: Wilcoxon test of each ablation mode vs the full
    AlphaGenome model (all at window_size == 23).

    Parameters
    ----------
    abl_df_raw : DataFrame
        Split-level AUCPR data for ablation modes.
        Must contain columns: mode, split, AUCPR.
    reference_mode : str
        The reference mode (default "AG").

    Returns
    -------
    DataFrame with columns: mode, p_value, padj
    """
    ref_data = abl_df_raw[abl_df_raw["mode"] == reference_mode].copy()
    results = []

    for mode, mode_data in abl_df_raw.groupby("mode"):
        if mode == reference_mode:
            continue
        combined = pd.concat([ref_data, mode_data])
        pivot = combined.pivot_table(index="split", columns="mode", values="AUCPR")
        if mode in pivot.columns and reference_mode in pivot.columns:
            stat, p = wilcoxon(pivot[mode], pivot[reference_mode])
            results.append({"mode": mode, "p_value": p})

    if not results:
        return pd.DataFrame(columns=["mode", "p_value", "padj"])
    df = pd.DataFrame(results)
    _, df["padj"], _, _ = multipletests(df["p_value"])
    return df


def calc_sig_window(pdf):
    """
    Figure 2: for each mode, Wilcoxon test of context sizes 32, 128, 512 nt
    against that *same* mode at 23 nt.  (i.e. AG_512 vs AG_23, not vs Base_23)

    Parameters
    ----------
    pdf : DataFrame
        Split-level AUCPR data with integer window_size column.
        Must contain columns: mode, window_size (int), split, AUCPR.

    Returns
    -------
    DataFrame with columns: mode, window_size, p_value, padj
    """
    wilcoxon_results = []

    for mode, mode_data in pdf.groupby("mode"):
        base_data = mode_data[mode_data["window_size"] == 23]
        if base_data.empty:
            continue

        for ws in [128, 512]:
            ws_data = mode_data[mode_data["window_size"] == ws]
            if ws_data.empty:
                continue
            combined = pd.concat([base_data, ws_data])
            pivot = combined.pivot_table(
                index="split", columns="window_size", values="AUCPR"
            )
            if 23 in pivot.columns and ws in pivot.columns:
                stat, p = wilcoxon(pivot[ws], pivot[23])
                wilcoxon_results.append(
                    {"mode": mode, "window_size": ws, "p_value": p}
                )

    if not wilcoxon_results:
        return pd.DataFrame(columns=["mode", "window_size", "p_value", "padj"])
    wdf = pd.DataFrame(wilcoxon_results)
    _, wdf["padj"], _, _ = multipletests(wdf["p_value"])
    return wdf


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------

def _apply_common_trace_style(fig):
    for trace in fig.data:
        trace.opacity = 1
        trace.marker.line.color = "black"
        trace.marker.line.width = 1
        if hasattr(trace, "error_y") and trace.error_y is not None:
            trace.error_y.thickness = 1


def _add_panel_labels(fig, labels_x, y=1.05, y_ref="paper"):
    """Add bold panel labels (a, b, c …) at given paper-space x positions."""
    for label, x in labels_x:
        fig.add_annotation(
            text=f"<b>{label}</b>",
            font=dict(size=PT8, family="Arial"),
            x=x, y=y,
            xanchor="left", yanchor="bottom",
            xref="paper", yref=y_ref,
            showarrow=False,
        )


# ---------------------------------------------------------------------------
# Figure 1 – three-panel bar chart
# ---------------------------------------------------------------------------

def make_figure1(agg_npdf, agg_pdf_23, abl_df, wilcoxon_fig1, wilcoxon_abl):
    """
    Three bar-chart panels side by side.

    a  No-pretrain comparison  (CRISPert / CRISCross × Pretraining / No pretraining)
    b  23 nt context only      (Base / Ex / AG, tested vs Base)
    c  Ablation studies        (AG / -ATAC / -Histone, tested vs AG)

    Significance brackets are drawn for all pairwise comparisons vs the
    reference bar (including n.s.).
    """
    fig = make_subplots(rows=1, cols=3, horizontal_spacing=0.12)

    # ── Panel a: no-pretrain ──────────────────────────────────────────────
    fig_npdf = px.bar(
        agg_npdf, x="model", y="mean", color="pretrain",
        barmode="group",
        category_orders={"model": ["CRISPert", "CRISCross"]},
        color_discrete_map={
            "No pretraining": COLORS["white"],
            "Pretraining":    COLORS["jaxgold"],
        },
    )
    for trace in fig_npdf.data:
        trace.update(legend="legend")
        fig.add_trace(trace, row=1, col=1)

    # ── Panel b: 23 nt, Base vs Ex vs AG (go.Bar to preserve x alignment) ─
    mode_colors = {
        "Base": COLORS["white"],
        "Ex":   COLORS["jaxpetrol"],
        "AG":   COLORS["jaxgold"],
    }
    ex_order = ["Base", "Ex", "AG"]
    for mode_name in ex_order:
        row_data = agg_pdf_23[agg_pdf_23["mode"] == mode_name]
        if row_data.empty:
            continue
        fig.add_trace(
            go.Bar(
                x=[mode_name],
                y=row_data["mean"].values,
                name=mode_name,
                marker=dict(color=mode_colors[mode_name]),
                showlegend=False,
                legendgroup=f"panel_b_{mode_name}",
            ),
            row=1, col=2,
        )

    # ── Panel c: ablation – distinct grey shades per bar ─────────────────
    abl_modes = ["AG", "-ATAC", "-Histone"]
    abl_colors = {
        "AG":       COLORS["seagrey"],   # darkest – the reference
        "-ATAC":    "#b2bec3",           # medium grey
        "-Histone": "#dfe6e9",           # lightest grey
    }
    for abl_mode in abl_modes:
        abl_row = abl_df[abl_df["mode"] == abl_mode]
        if abl_row.empty:
            continue
        fig.add_trace(
            go.Bar(
                x=[abl_mode],
                y=abl_row["mean"].values,
                name=abl_mode,
                marker=dict(color=abl_colors[abl_mode]),
                showlegend=False,
                legendgroup=f"panel_c_{abl_mode}",
            ),
            row=1, col=3,
        )

    # ── x-axis labels ─────────────────────────────────────────────────────
    nopretrain_modes = ["CRISPert", "CRISCross"]
    fig.update_xaxes(
        categoryorder="array", categoryarray=nopretrain_modes, row=1, col=1
    )
    fig.update_xaxes(
        categoryorder="array", categoryarray=ex_order,
        tickvals=ex_order,
        ticktext=["Sequence only", "Experimental", "AlphaGenome"],
        row=1, col=2,
    )
    fig.update_xaxes(
        categoryorder="array", categoryarray=abl_modes,
        tickvals=abl_modes,
        ticktext=["Full AlphaGenome", "-ATAC", "-Histone"],
        row=1, col=3,
    )

    # ── Bar value annotations ──────────────────────────────────────────────
    add_bar_values(
        fig=fig, pdf=agg_npdf, modes=nopretrain_modes,
        row=1, col=1, cat_col="pretrain", x_val="model", offset_x=0,
    )
    add_bar_values(
        fig=fig, pdf=agg_pdf_23, modes=ex_order,
        row=1, col=2, cat_col="window_size", x_val="mode", offset_x=0,
    )
    add_bar_values(
        fig=fig, pdf=abl_df, modes=abl_modes,
        row=1, col=3, cat_col="_none_", x_val="mode", offset_x=0,
    )

    # ── Significance brackets – panel b (Ex, AG vs Base) ──────────────────
    # x positions: Base=0, Ex=1, AG=2
    # Anchor off a fixed y-floor (axis max minus headroom) so brackets are
    # positioned consistently regardless of which bar is tallest.
    comps_b = []
    for mode_name in ["Ex", "AG"]:
        prow = wilcoxon_fig1[wilcoxon_fig1["mode"] == mode_name]
        if prow.empty:
            continue
        label = stars_label(prow["padj"].values[0])
        x_test = ex_order.index(mode_name)
        comps_b.append((0, x_test, label))  # reference Base is at x=0

    if comps_b:
        # Use a fixed clearance above the tallest bar value text (rotated
        # annotations extend ~0.015 above bar top in data coords at this scale)
        y_max_b = agg_pdf_23["mean"].max()
        add_significance_brackets(
            fig, comps_b,
            y_start=y_max_b + 0.02,
            y_step=0.016,
            row=1, col=2, ncols=3,
        )

    # ── Significance brackets – panel c (-ATAC, -Histone vs AG) ───────────
    # x positions: AG=0, -ATAC=1, -Histone=2
    comps_c = []
    for mode_name in ["-ATAC", "-Histone"]:
        prow = wilcoxon_abl[wilcoxon_abl["mode"] == mode_name]
        if prow.empty:
            continue
        label = stars_label(prow["padj"].values[0])
        x_test = abl_modes.index(mode_name)
        comps_c.append((0, x_test, label))  # reference AG is at x=0

    if comps_c:
        y_max_c = abl_df["mean"].max()
        add_significance_brackets(
            fig, comps_c,
            y_start=y_max_c + 0.02,
            y_step=0.016,
            row=1, col=3, ncols=3,
        )

    # ── Layout ────────────────────────────────────────────────────────────
    fig.update_layout(
        template="simple_white_custom",
        height=250, width=WIDTH,
        bargroupgap=0.2, bargap=0.3,
        margin=dict(l=60, t=25),
        legend=dict(
            title=dict(text=""),
            x=0., y=0.85, orientation="h",
            xanchor="left", yanchor="bottom",
        ),
    )
    _apply_common_trace_style(fig)
    fig.update_xaxes(title=dict(text=None))
    fig.update_yaxes(title=dict(text="AUC-PR"))
    fig.update_yaxes(range=[0, 1.15],  row=1, col=1)
    fig.update_yaxes(range=[0.7, 0.92], row=1, col=2)
    fig.update_yaxes(range=[0.7, 0.92], row=1, col=3)

    _add_panel_labels(fig, [("a", -0.04), ("b", 0.34), ("c", 0.70)], y=0.99)

    return fig


# ---------------------------------------------------------------------------
# Figure 2 – context-size comparison
# ---------------------------------------------------------------------------

def make_figure2(agg_pdf, wilcoxon_window_df):
    """
    Three bar-chart panels, one per model type, showing performance at
    context sizes 23, 32, 128 and 512 nt.

    For each panel the significance brackets compare every non-reference
    context size against that *same* model's 23 nt result
    (e.g. AG_512 vs AG_23, not AG_512 vs Base_23).

    a  Base model      (Sequence only)
    b  Ex model        (Experimental)
    c  AG model        (AlphaGenome)
    """
    custom_colors = {
        "23":  COLORS["jaxgold"],
        "128": COLORS["white"],
        "512": COLORS["jaxpetrol"],
    }
    window_order = ["23", "128", "512"]   # categorical strings
    epi_modes = ["Base", "Ex", "AG"]
    mode_labels = {
        "Base": "Sequence only",
        "Ex":   "Experimental",
        "AG":   "AlphaGenome",
    }

    fig = make_subplots(
        rows=1, cols=3,
        horizontal_spacing=0.10,
        shared_yaxes=True,
        subplot_titles=[mode_labels[m] for m in epi_modes],
    )
    fig.update_annotations(font=dict(size=PT7))

    for col_idx, mode in enumerate(epi_modes, start=1):
        # Convert window_size to string category so bars are evenly spaced
        mode_data = (
            agg_pdf[agg_pdf["mode"] == mode]
            .copy()
            .assign(window_size=lambda d: d["window_size"].astype(int).astype(str))
            .sort_values("window_size")
        )

        # One go.Bar per context size → stable colours, no alignment issues
        for ws in window_order:
            ws_row = mode_data[mode_data["window_size"] == ws]
            if ws_row.empty:
                continue
            fig.add_trace(
                go.Bar(
                    x=[ws],
                    y=ws_row["mean"].values,
                    name=f"{ws} nt",
                    marker=dict(color=custom_colors[ws]),
                    legendgroup=ws,
                    showlegend=(col_idx == 1),  # legend entry once only
                ),
                row=1, col=col_idx,
            )

        # ── Significance brackets vs same-mode 23 nt ──────────────────────
        # x positions: 23=0, 128=1, 512=2
        # Use a fixed global floor (tallest bar across all modes + text
        # clearance) so brackets sit at the same height in every panel.
        if not wilcoxon_window_df.empty:
            mode_stats = (
                wilcoxon_window_df[wilcoxon_window_df["mode"] == mode]
                .copy()
                .assign(window_size=lambda d: d["window_size"].astype(str))
            )
            comps = []
            for ws in ["128", "512"]:
                prow = mode_stats[mode_stats["window_size"] == ws]
                if prow.empty:
                    continue
                label = stars_label(prow["padj"].values[0])
                x_test = window_order.index(ws)
                comps.append((0, x_test, label))  # 23 nt is at x=0

            if comps:
                # Rotated text extends ~0.008 above bar top; add extra buffer
                y_max = mode_data["mean"].max()
                add_significance_brackets(
                    fig, comps,
                    y_start=y_max + 0.02,
                    y_step=0.010,
                    row=1, col=col_idx, ncols=3,
                )

        # ── Bar value annotations ─────────────────────────────────────────
        add_bar_values(
            fig=fig, pdf=mode_data, modes=window_order,
            row=1, col=col_idx,
            cat_col="_none_", x_val="window_size", offset_x=0,
        )

        # ── x-axis ────────────────────────────────────────────────────────
        fig.update_xaxes(
            categoryorder="array", categoryarray=window_order,
            tickvals=window_order,
            ticktext=["23 nt", "128 nt", "512 nt"],
            title=dict(text=None),
            row=1, col=col_idx,
        )

    # ── Layout ────────────────────────────────────────────────────────────
    fig.update_layout(
        template="simple_white_custom",
        height=300, width=WIDTH,
        bargap=0.3,
        margin=dict(r=90, t=45),
        legend=dict(
            title=dict(text="Context size"),
            x=1.02, y=0.5,
            xanchor="left", yanchor="middle",
        ),
    )
    _apply_common_trace_style(fig)
    # Extend y-range to give brackets and subplot titles room to breathe
    fig.update_yaxes(range=[0.7, 0.85])
    fig.update_yaxes(title=dict(text="AUC-PR"), col=1)

    _add_panel_labels(fig, [("a", -0.05), ("b", 0.35), ("c", 0.70)], y=1.05)

    return fig


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main(df):
    df.loc[df["mode"].isna(), "mode"] = "Base"
    epi_modes = ["Base", "Ex", "AG"]

    df = df.groupby(
        ["mode", "window_size", "split", "pretrain", "model"]
    )["AUCPR"].mean().reset_index()
    df["mode"] = df["mode"].map(lambda x: NAME_FIX.get(x, x))
    df["name"] = df["mode"] + "_" + df["window_size"].astype(str)

    # ── epi subset (Base / Ex / AG, all window sizes) ─────────────────────
    pdf = df[df["mode"].isin(epi_modes)].copy()

    # ── no-pretrain subset ────────────────────────────────────────────────
    no_pretrain_comp = ["NoPretrain", "Base", "CRISPert no pretrain", "CRISPert original"]
    npdf = df[df["mode"].isin(no_pretrain_comp) & (df["window_size"] == 23)].copy()
    npdf["window_size"] = npdf["window_size"].astype("category")
    agg_npdf = (
        npdf.groupby(["pretrain", "model"])["AUCPR"]
        .agg(["mean", "std"])
        .reset_index()
    )
    agg_npdf["pretrain"] = agg_npdf["pretrain"].apply(
        lambda x: "Pretraining" if x else "No pretraining"
    )

    # ── Fig 1 panel b: Ex_23 and AG_23 vs Base_23 ─────────────────────────
    pdf_23_raw = pdf[pdf["window_size"] == 23].copy()
    wilcoxon_fig1 = calc_sig_fig1(pdf_23_raw, reference_mode="Base")
    print("Fig 1 panel b – modes at 23 nt vs Base_23:")
    print(wilcoxon_fig1)

    # ── Fig 1 panel c: ablation at 23 nt vs AG_23 ─────────────────────────
    abl_comp = ["AG", "-ATAC", "-Histone"]
    abl_df_raw = df[df["mode"].isin(abl_comp) & (df["window_size"] == 23)].copy()
    wilcoxon_abl = calc_sig_ablation(abl_df_raw, reference_mode="AG")
    print("\nFig 1 panel c – ablation vs AG_23:")
    print(wilcoxon_abl)
    abl_df = (
        abl_df_raw.groupby(["mode"])["AUCPR"]
        .agg(["mean", "std"])
        .reset_index()
    )

    # ── Aggregate bar data ────────────────────────────────────────────────
    pdf["window_size"] = pdf["window_size"].astype("category")
    agg_pdf = (
        pdf.groupby(["window_size", "mode"])["AUCPR"]
        .agg(["mean", "std"])
        .reset_index()
    )
    agg_pdf_23 = agg_pdf[agg_pdf["window_size"] == 23].copy()

    # ── Fig 2: each mode's 32/128/512 nt vs its own 23 nt baseline ────────
    pdf_int = pdf.copy()
    pdf_int["window_size"] = pdf_int["window_size"].astype(int)
    wilcoxon_window_df = calc_sig_window(pdf_int)
    print("\nFig 2 – context-size comparisons (each mode vs its own 23 nt):")
    print(wilcoxon_window_df)

    # ── Build and save figures ────────────────────────────────────────────
    fig1 = make_figure1(agg_npdf, agg_pdf_23, abl_df, wilcoxon_fig1, wilcoxon_abl)
    fig1.write_html("Figures/CrossAttn_Fig1.html")
    fig1.write_image(
        "Figures/CrossAttn_Fig1.svg",
        width=fig1.layout.width, height=fig1.layout.height,
    )
    print("\nFigure 1 saved.")

    fig2 = make_figure2(agg_pdf, wilcoxon_window_df)
    fig2.write_html("Figures/CrossAttn_Fig2.html")
    fig2.write_image(
        "Figures/CrossAttn_Fig2.svg",
        width=fig2.layout.width, height=fig2.layout.height,
    )
    print("Figure 2 saved.")


if __name__ == "__main__":
    file = "../Results/FinalSummary.tsv"
    df = pd.read_csv(file, sep="\t")
    main(df)
