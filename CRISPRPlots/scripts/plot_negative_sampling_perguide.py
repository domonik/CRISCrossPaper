import pandas as pd
from plotly_template import COLORS, SINGLE_COL, PT6
import plotly.graph_objects as go


def main():
    file = "AllExperiments_PerGuide.tsv"
    df = pd.read_csv(file, sep="\t")

    # Reference dataset (CrossCellTypeTest) plus the negative-sampling runs.
    # CrossCellTypeCrossNegative{10,20,30,40,50} are re-runs of the same
    # subsampling setup with a different random seed (not different negative
    # ratios); CrossCellTypeCrossNegativeHard uses a distinct hard-negative
    # sampling strategy.
    dataset_to_group = {
        "CrossCellTypeTest": "Reference",
        "CrossCellTypeCrossNegative10": "Seed10",
        "CrossCellTypeCrossNegative20": "Seed20",
        "CrossCellTypeCrossNegative30": "Seed30",
        "CrossCellTypeCrossNegative40": "Seed40",
        "CrossCellTypeCrossNegative50": "Seed50",
        "CrossCellTypeCrossNegativeHard": "Hard",
    }
    df = df[df["dataset"].isin(dataset_to_group)].copy()
    df["group"] = df["dataset"].map(dataset_to_group)

    # ── Average per guide (mean over seeds within a run) ────────────
    # One aucpr value per (group, condition, GuideID).
    avg_df = df.groupby(["group", "condition", "GuideID"])["aucpr"].mean().reset_index()

    row_groups = ["Reference", "Seed10", "Seed20", "Seed30", "Seed40", "Seed50", "Hard"]

    # x-axis names each dataset/group (Reference / Reseed 1-5 / Hard
    # Negatives) — the -AG vs +AG condition is explained via bar color
    # in the legend instead.
    group_display_names = {
        "Reference": "Reference",
        "Seed10": "Reseed 1",
        "Seed20": "Reseed 2",
        "Seed30": "Reseed 3",
        "Seed40": "Reseed 4",
        "Seed50": "Reseed 5",
        "Hard": "Hard<br>Negatives",
    }
    group_order = [group_display_names[g] for g in row_groups]

    # Reference gets one diagonal, the reseeded runs get the opposite
    # diagonal, and Hard negatives keep their own distinct pattern.
    bar_hatches = {"Reference": "/", "Hard": "x"}
    for g in ["Seed10", "Seed20", "Seed30", "Seed40", "Seed50"]:
        bar_hatches[g] = "\\"
    hatch_by_group_order = [bar_hatches[g] for g in row_groups]

    # ── Aggregate for bar chart (mean over the per-guide values) ──
    agg_df = avg_df.groupby(["group", "condition"])["aucpr"].agg(["mean", "std"]).reset_index()

    # ── Build figure ─────────────────────────────────────────────────
    # Exactly two traces — one per condition — each spanning all 7
    # dataset/group categories on the x-axis. This is what makes
    # barmode="group" place a single -AG/+AG pair side by side per
    # category instead of splitting the width across many single-bar
    # traces.
    fig = go.Figure()

    condition_colors = {"-AG": COLORS["seagrey"], "+AG": COLORS["jaxgold"]}

    # barmode="group" sizes bars by the number of distinct offsetgroups on
    # the subplot, counting every Bar trace present — including ones with
    # no real x data (like the fake legend swatches below). Pin the two
    # real traces to fixed offsetgroups so only these two define the
    # layout, then reuse the same offsetgroup ids for the fake traces so
    # they don't add extra (empty) slots.
    offsetgroup_by_condition = {"-AG": "0", "+AG": "1"}

    for condition in ["-AG", "+AG"]:
        cdf = agg_df[agg_df["condition"] == condition].set_index("group")
        y_vals = [cdf.loc[g, "mean"] if g in cdf.index else None for g in row_groups]

        fig.add_trace(
            go.Bar(
                x=group_order,
                y=y_vals,
                marker=dict(
                    color=condition_colors[condition],
                    line=dict(color="black", width=1),
                    pattern=dict(shape=hatch_by_group_order),
                ),
                text=[f"{v:.3f}" if v is not None else "" for v in y_vals],
                textposition="outside",
                textangle=90,
                textfont=dict(size=PT6, color="black"),
                width=0.4,
                offsetgroup=offsetgroup_by_condition[condition],
                showlegend=False,
            )
        )


    fig.update_xaxes(
        categoryorder="array",
        categoryarray=group_order,
    )

    # ── Legend entries (invisible fake traces, not part of the plot) ──
    # These have no real x data, so they draw nothing — but they still
    # need to sit in an existing offsetgroup ("0"), not a new one, or
    # they'd steal width from the real bars again (see comment above).
    fake_offsetgroup = offsetgroup_by_condition["-AG"]

    # Two solid-color swatches explain the -AG / +AG condition — these go
    # in the default legend ("legend").
    fig.add_trace(
        go.Bar(
            x=[None], y=[None],
            marker=dict(color=COLORS["seagrey"], line=dict(color="black", width=1)),
            name="Sequence only",
            offsetgroup=fake_offsetgroup,
            legend="legend",
        )
    )
    fig.add_trace(
        go.Bar(
            x=[None], y=[None],
            marker=dict(color=COLORS["jaxgold"], line=dict(color="black", width=1)),
            name="Sequence & AlphaGenome ATAC-seq",
            offsetgroup=fake_offsetgroup,
            legend="legend",
        )
    )
    # ...and three black swatches with the fill patterns explain group
    # identity — these go in a second, separate legend ("legend2") so the
    # two rows each stay short enough to render horizontally instead of
    # wrapping.
    fig.add_trace(
        go.Bar(
            x=[None], y=[None],
            marker=dict(color=COLORS["black"], line=dict(color="black", width=1), pattern=dict(shape="/")),
            name="Reference",
            offsetgroup=fake_offsetgroup,
            legend="legend2",
        )
    )
    fig.add_trace(
        go.Bar(
            x=[None], y=[None],
            marker=dict(color=COLORS["black"], line=dict(color="black", width=1), pattern=dict(shape="\\")),
            name="Reseed 1-5",
            offsetgroup=fake_offsetgroup,
            legend="legend2",
        )
    )
    fig.add_trace(
        go.Bar(
            x=[None], y=[None],
            marker=dict(color=COLORS["black"], line=dict(color="black", width=1), pattern=dict(shape="x")),
            name="Hard Negatives",
            offsetgroup=fake_offsetgroup,
            legend="legend2",
        )
    )

    # ── Layout ──────────────────────────────────────────────────────
    fig.update_layout(
        barmode="group",
        template="simple_white_custom",
        height=300,
        width=SINGLE_COL,
        margin=dict(t=20, l=30, r=10, b=90),
        legend=dict(orientation="h", y=-0.12, x=-0.1, xanchor="left"),
        legend2=dict(orientation="h", y=-0.22, x=-0.1, xanchor="left"),
    )

    fig.update_yaxes(title_text="AUC-PR", range=[0, 1.15])

    fig.write_html("Figures/CRISPRAT_NegativeSampling_PerGuide.html")
    fig.write_image("Figures/CRISPRAT_NegativeSampling_PerGuide.svg", width=fig.layout.width, height=fig.layout.height)
    print("Saved Figures/CRISPRAT_NegativeSampling_PerGuide.html and Figures/CRISPRAT_NegativeSampling_PerGuide.svg")


if __name__ == "__main__":
    main()