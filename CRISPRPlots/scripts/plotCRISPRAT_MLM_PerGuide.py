import pandas as pd
from statsmodels.stats.multitest import multipletests
from scipy.stats import wilcoxon
from plotly_template import COLORS, SINGLE_COL, PT6
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np


def add_significance_brackets(
    fig,
    comparisons,
    y_start,
    y_step,
    xref,
    yref,
    font_size_stars=12,
    tick_frac=0.35,
):
    """
    Draw horizontal significance brackets above grouped bars. Brackets whose
    x-ranges don't overlap share the same height; only overlapping brackets
    are stacked (interval-scheduling levels), so unrelated comparisons don't
    drift upward for no reason.
    """
    tick_h = y_step * tick_frac
    sorted_comps = sorted(comparisons, key=lambda c: (min(c[0], c[1]), max(c[0], c[1])))

    level_ends = []
    placements = []
    for x0, x1, label in sorted_comps:
        lo, hi = min(x0, x1), max(x0, x1)
        for level, end in enumerate(level_ends):
            if lo >= end:
                level_ends[level] = hi
                placements.append((level, x0, x1, label))
                break
        else:
            level_ends.append(hi)
            placements.append((len(level_ends) - 1, x0, x1, label))

    for level, x0, x1, label in placements:
        y = y_start + level * y_step

        fig.add_shape(
            type="line", x0=x0, x1=x1, y0=y, y1=y,
            xref=xref, yref=yref,
            line=dict(color="black", width=1),
        )
        fig.add_shape(
            type="line", x0=x0, x1=x0, y0=y - tick_h, y1=y,
            xref=xref, yref=yref,
            line=dict(color="black", width=1),
        )
        fig.add_shape(
            type="line", x0=x1, x1=x1, y0=y - tick_h, y1=y,
            xref=xref, yref=yref,
            line=dict(color="black", width=1),
        )
        is_stars = "*" in label
        fig.add_annotation(
            x=(x0 + x1) / 2, y=y - 0.06 * is_stars,
            yanchor="bottom", xanchor="center",
            text=label, showarrow=False,
            xref=xref, yref=yref,
            font=dict(size=font_size_stars if is_stars else PT6),
        )


def stars_label(p):
    """Returns star string or 'n.s.' – always produces a label."""
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    return "n.s."


def main():
    file = "AllExperiments_PerGuide.tsv"
    df = pd.read_csv(file, sep="\t")

    # ── Row 1: LOSO fine-tuning (T-Cell / K562) ─────────────────────
    # CRISPRAT vs Dataset-specific MLM pretraining, compared per guide.
    loso_dataset_to_group = {
        "LOSOFineTuneCRISPRATTCell": "Arti",
        "LOSOFineTuneMLMTCell": "DatasetMLM",
        "LOSOFineTuneCRISPRATK562": "Arti",
        "LOSOFineTuneMLMK562": "DatasetMLM",
    }
    loso_dataset_to_cell = {
        "LOSOFineTuneCRISPRATTCell": "T-Cell",
        "LOSOFineTuneMLMTCell": "T-Cell",
        "LOSOFineTuneCRISPRATK562": "K562",
        "LOSOFineTuneMLMK562": "K562",
    }
    loso_df = df[df["dataset"].isin(loso_dataset_to_group)].copy()
    loso_df["group"] = loso_df["dataset"].map(loso_dataset_to_group)
    loso_df["cell"] = loso_df["dataset"].map(loso_dataset_to_cell)

    # ── Average per guide (mean over seeds) ─────────────────────────
    loso_avg_df = loso_df.groupby(["cell", "group", "GuideID"])["aucpr"].mean().reset_index()

    loso_cells = ["T-Cell", "K562"]
    loso_groups = ["Arti", "DatasetMLM"]
    # Hatch encodes pretraining strategy only (consistent across all bars):
    # CRISPRAT-pretrained ("Arti" and "NoCrossAttn", which is CRISPRAT-pretrained
    # with cross-attention replaced by simple embedding addition) get the same
    # hatch; Dataset-specific MLM pretraining is filled (no hatch).
    bar_hatches = {"Arti": "/", "DatasetMLM": "", "NoCrossAttn": "/"}

    # ── Paired Wilcoxon signed-rank test: CRISPRAT vs Dataset-specific MLM ──
    loso_pvals = []
    for cell in loso_cells:
        arti = loso_avg_df[(loso_avg_df["cell"] == cell) & (loso_avg_df["group"] == "Arti")].set_index("GuideID")["aucpr"]
        mlm = loso_avg_df[(loso_avg_df["cell"] == cell) & (loso_avg_df["group"] == "DatasetMLM")].set_index("GuideID")["aucpr"]
        arti, mlm = arti.align(mlm, join="inner")
        if len(arti) > 0:
            stat, p = wilcoxon(arti, mlm)
            loso_pvals.append(p)
        else:
            loso_pvals.append(np.nan)

    loso_pvals_array = np.array(loso_pvals, dtype=float)
    loso_mask = ~np.isnan(loso_pvals_array)
    loso_padj = np.full_like(loso_pvals_array, np.nan)
    loso_padj[loso_mask] = multipletests(loso_pvals_array[loso_mask], method="bonferroni")[1]

    print("=== Paired Wilcoxon signed-rank test (CRISPRAT vs Dataset-specific MLM, per guide, LOSO) ===")
    for cell, p, pa in zip(loso_cells, loso_pvals, loso_padj):
        print(f"{cell}: p={p:.2e}, padj={pa:.2e}")

    # ── Aggregate for bar chart ──────────────────────────────────────
    loso_agg_df = loso_avg_df.groupby(["cell", "group"])["aucpr"].agg(["mean", "std"]).reset_index()

    # ── Row 2: CrossCellType comparison ─────────────────────────────
    dataset_to_group = {
        "CrossCellTypeTest": "Arti",
        "CrossCellTypeTestMLMFull": "DatasetMLM",
        "CrossCellTypeCrossAttentionAblation": "NoCrossAttn",
    }
    cc_df = df[df["dataset"].isin(dataset_to_group)].copy()
    cc_df["group"] = cc_df["dataset"].map(dataset_to_group)

    # Average per guide (mean over seeds). One aucpr value per (group, condition, GuideID).
    avg_df = cc_df.groupby(["group", "condition", "GuideID"])["aucpr"].mean().reset_index()

    # Arti and NoCrossAttn are adjacent on purpose: both are CRISPRAT-pretrained
    # (shared hatch), differing only in architecture, so the pair sits next to
    # each other and gets a shared "CRISPRAT pretraining" bracket underneath
    # (see below). DatasetMLM is the lone arm testing a different pretraining
    # strategy and is named fully on its own.
    row2_groups = ["Arti", "NoCrossAttn", "DatasetMLM"]
    row2_conditions = ["-AG", "+AG"]

    bar_order = [f"{g}_{c}" for g in row2_groups for c in row2_conditions]
    # Second line names the pretraining strategy: Arti and NoCrossAttn are
    # both CRISPRAT-pretrained (shared hatch, see above), DatasetMLM uses
    # Dataset-specific MLM pretraining instead.
    group_display = {
        "Arti": "Cross-Attention<br>CRISPRAT",
        "NoCrossAttn": "No Cross-Attention<br>CRISPRAT",
        "DatasetMLM": "Cross-Attention<br>MLM pretraining",
    }
    condition_display = {"-AG": "Sequence", "+AG": "Sequence<br>&<br>ATAC"}

    # ── Paired Wilcoxon signed-rank test: -AG vs +AG, within each group ──
    pvals = []
    for group in row2_groups:
        minus_ag = avg_df[(avg_df["group"] == group) & (avg_df["condition"] == "-AG")].set_index("GuideID")["aucpr"]
        plus_ag = avg_df[(avg_df["group"] == group) & (avg_df["condition"] == "+AG")].set_index("GuideID")["aucpr"]
        minus_ag, plus_ag = minus_ag.align(plus_ag, join="inner")
        if len(minus_ag) > 0:
            stat, p = wilcoxon(minus_ag, plus_ag)
            pvals.append(p)
        else:
            pvals.append(np.nan)

    pvals_array = np.array(pvals, dtype=float)
    mask = ~np.isnan(pvals_array)
    padj = np.full_like(pvals_array, np.nan)
    padj[mask] = multipletests(pvals_array[mask], method="bonferroni")[1]

    print("=== Paired Wilcoxon signed-rank test (-AG vs +AG, per guide) ===")
    for group, p, pa in zip(row2_groups, pvals, padj):
        print(f"{group}: p={p:.2e}, padj={pa:.2e}")

    # ── Aggregate for bar chart (mean/std over the 12 per-guide values) ──
    agg_df = avg_df.groupby(["group", "condition"])["aucpr"].agg(["mean", "std"]).reset_index()
    agg_df["bar"] = agg_df["group"] + "_" + agg_df["condition"]

    # ── Build figure ─────────────────────────────────────────────────
    fig = make_subplots(
        rows=2, cols=2,
        specs=[
            [{"type": "xy"}, {"type": "xy"}],
            [{"type": "xy", "colspan": 2}, None],
        ],
        vertical_spacing=0.08,
        horizontal_spacing=0.1,
    )

    # ── Row 1, Col 1: T-Cell LOSO ───────────────────────────────────
    for group in loso_groups:
        bar = loso_agg_df[(loso_agg_df["cell"] == "T-Cell") & (loso_agg_df["group"] == group)]
        if bar.empty:
            continue
        fig.add_trace(
            go.Bar(
                x=[group],
                y=[bar["mean"].values[0]],
                marker=dict(
                    color=COLORS["seagrey"],
                    line=dict(color="black", width=1),
                    pattern=dict(shape=bar_hatches[group]),
                ),
                showlegend=False,
                width=0.32,
            ),
            row=1, col=1,
        )

    fig.update_xaxes(
        categoryorder="array",
        categoryarray=loso_groups,
        tickvals=loso_groups,
        ticktext=["Sequence", "Sequence"],
        row=1, col=1,
    )

    # ── Row 1, Col 2: K562 LOSO ──────────────────────────────────────
    for group in loso_groups:
        bar = loso_agg_df[(loso_agg_df["cell"] == "K562") & (loso_agg_df["group"] == group)]
        if bar.empty:
            continue
        fig.add_trace(
            go.Bar(
                x=[group],
                y=[bar["mean"].values[0]],
                marker=dict(
                    color=COLORS["seagrey"],
                    line=dict(color="black", width=1),
                    pattern=dict(shape=bar_hatches[group]),
                ),
                showlegend=False,
                width=0.32,
            ),
            row=1, col=2,
        )

    fig.update_xaxes(
        categoryorder="array",
        categoryarray=loso_groups,
        tickvals=loso_groups,
        ticktext=["Sequence", "Sequence"],
        row=1, col=2,
    )

    # Significance brackets + bar value annotations for row 1
    loso_xrefs = {"T-Cell": ("x", "y"), "K562": ("x2", "y2")}
    for cell, pa in zip(loso_cells, loso_padj):
        xref, yref = loso_xrefs[cell]
        cell_agg = loso_agg_df[loso_agg_df["cell"] == cell]

        if not np.isnan(pa):
            y_max = cell_agg["mean"].max()
            add_significance_brackets(
                fig, [(0, 1, stars_label(pa))],
                y_start=y_max + 0.15,
                y_step=0.08,
                xref=xref, yref=yref,
            )

        for group in loso_groups:
            bar = cell_agg[cell_agg["group"] == group]
            if bar.empty:
                continue
            fig.add_annotation(
                x=group, y=bar["mean"].values[0],
                text=f"{bar['mean'].values[0]:.3f}",
                showarrow=False, xref=xref, yref=yref,
                yanchor="bottom", xanchor="center",
                font=dict(size=PT6, color="black"),
            )

    # ── Row 2 (spanned): CrossCellType comparison ───────────────────
    # Multicategory x-axis: inner tick = condition (Sequence / Sequence & ATAC),
    # outer grouping label underneath = which experiment the pair belongs to.
    for bar_key in bar_order:
        group, condition = bar_key.split("_")
        row = agg_df[agg_df["bar"] == bar_key]
        if row.empty:
            continue
        fig.add_trace(
            go.Bar(
                x=[[group_display[group]], [condition_display[condition]]],
                y=[row["mean"].values[0]],
                marker=dict(
                    color=COLORS["jaxgold"] if condition == "+AG" else COLORS["seagrey"],
                    line=dict(color="black", width=1),
                    pattern=dict(shape=bar_hatches[group]),
                ),
                showlegend=False,
                width=0.4,
            ),
            row=2, col=1,
        )

    # Significance brackets: -AG vs +AG within each group
    comps = []
    for i, (group, pa) in enumerate(zip(row2_groups, padj)):
        if np.isnan(pa):
            continue
        x0 = i * 2
        x1 = i * 2 + 1
        comps.append((x0, x1, stars_label(pa)))

    if comps:
        y_max = agg_df["mean"].max()
        add_significance_brackets(
            fig, comps,
            y_start=y_max + 0.15,
            y_step=0.08,
            xref="x3", yref="y3",
        )

    # Bar value annotations (positioned by category index, matching the
    # brackets above)
    for i, bar_key in enumerate(bar_order):
        row = agg_df[agg_df["bar"] == bar_key]
        if row.empty:
            continue
        fig.add_annotation(
            x=i, y=row["mean"].values[0],
            text=f"{row['mean'].values[0]:.3f}",
            showarrow=False, xref="x3", yref="y3",
            yanchor="bottom", xanchor="center",
            font=dict(size=PT6, color="black"),
        )

    # ── Legend entries (invisible bars) ──────────────────────────────
    # Hatch encodes pretraining strategy only; the "No Cross-Attention" group
    # is identified via its x-axis label instead (it shares CRISPRAT
    # pretraining, hence the shared hatch with the CRISPRAT bars).
    fig.add_trace(
        go.Bar(
            x=[None], y=[None],
            marker=dict(
                color=COLORS["black"],
                line=dict(color="black", width=1),
                pattern=dict(shape="/"),
            ),
            name="CRISPRAT",
        )
    )
    fig.add_trace(
        go.Bar(
            x=[None], y=[None],
            marker=dict(
                color=COLORS["black"],
                line=dict(color="black", width=1),
                pattern=dict(shape=""),
            ),
            name="Dataset-specific MLM pretraining",
        )
    )

    # ── Layout ──────────────────────────────────────────────────────
    fig.update_layout(
        barmode="group",
        template="simple_white_custom",
        height=400,
        width=SINGLE_COL,
        margin=dict(t=20, l=30, r=20, b=80),
        legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center"),
    )

    fig.update_yaxes(title_text="AUC-PR", range=[0, 1.15], row=1, col=1)
    fig.update_yaxes(title_text="AUC-PR", range=[0, 1.15], row=1, col=2)
    fig.update_yaxes(title_text="AUC-PR", range=[0, 1.15], row=2, col=1)

    fig.add_annotation(
        text="T-Cell<br>(leave one out)",
        x=0.5, xanchor="center", xref="x domain",
        y=1.1, yref="y domain",
        showarrow=False,
    )
    fig.add_annotation(
        text="K562<br>(leave one out)",
        x=0.5, xanchor="center", xref="x2 domain",
        y=1.1, yref="y2 domain",
        showarrow=False,
    )
    fig.add_annotation(
        text="K562<br>(Cross cell type)",
        x=0.5, xanchor="center", xref="x3 domain",
        y=1.0, yref="y3 domain",
        showarrow=False,
    )

    fig.write_html("Figures/CRISPRAT_MLM_PerGuide.html")
    fig.write_image("Figures/CRISPRAT_MLM_PerGuide.svg", width=fig.layout.width, height=fig.layout.height)
    print("Saved Figures/CRISPRAT_MLM_PerGuide.html and Figures/CRISPRAT_MLM_PerGuide.svg")


if __name__ == "__main__":
    main()
