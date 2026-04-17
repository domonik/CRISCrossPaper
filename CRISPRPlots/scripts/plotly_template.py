import plotly.graph_objects as go
import plotly.io as pio
import re

PT7 = 9.33333333333
PT8 = 10.666666666
PT6 = 8
PT5 = 6.66666666666

COLORS = {
    "balpurple": "rgb(26, 25, 95)",
    "balgold": "rgb(158, 124, 12)",
    "balred": "rgb(198, 12, 14)",
    "lardarkgold": "rgb(255, 130, 0)",
    "black": "rgb(0, 0, 0)",
    "white": "rgb(255, 255, 255)",
    "laryellow": "rgb(255, 209, 0)",
    "jaxgold": "rgb(215, 162, 42 )",
    "jaxpetrol": "rgb(0, 103, 120)",
    "vikpurple": "rgb(79, 38, 131)",
    "bufblue": "rgb(0, 51, 141)",
    "bufred": "rgb(198, 12, 48)",
    "miablue": "rgb(0, 142, 151)",
    "miaorange": "rgb(252, 76, 2)",
    "seagrey": "rgb(165, 172, 175)",
    "seagreen": "rgb(105, 190, 40)",
    # Categorical colors for discrete features
    "cat1": "rgb(79, 38, 131)",
    "cat2": "rgb(0, 51, 141)",
    "cat3": "rgb(0, 142, 151)",
    "cat4": "rgb(105, 190, 40)",
    "cat5": "rgb(198, 12, 48)",
    "cat6": "rgb(252, 76, 2)",
    "cat7": "rgb(198, 12, 48)",
    "cat8": "rgb(227, 119, 194)",
    "cat9": "rgb(127, 127, 127)",
    "cat10": "rgb(188, 189, 34)",
    "cat11": "rgb(23, 190, 207)",
}
WIDTH = 691
SINGLE_COL = 336


def rgb_to_rgba(color_str, opacity):
    """
    Convert a Plotly-style 'rgb(r,g,b)' or 'rgba(r,g,b,a)' string
    into 'rgba(r,g,b,opacity)'.

    Parameters
    ----------
    color_str : str
        e.g. 'rgb(31,119,180)' or 'rgba(31,119,180,1)'
    opacity : float
        Value between 0 and 1

    Returns
    -------
    str
        'rgba(r,g,b,opacity)'
    """
    match = re.match(r"rgba?\(([^)]+)\)", color_str.strip())
    if not match:
        raise ValueError("Input must be an rgb(...) or rgba(...) string")

    parts = match.group(1).split(",")
    r, g, b = [int(p.strip()) for p in parts[:3]]

    return f"rgba({r},{g},{b},{opacity})"

def register_template():
    base = pio.templates["plotly_white"]

    pio.templates["simple_white_custom"] = go.layout.Template(
        layout={
            **base.layout.to_plotly_json(),
            "width": WIDTH,
            "xaxis": {
                "title_font": {"size": PT6},
                "tickfont": {"size": PT5},
                "color": "black",
                "title_standoff": 5,
                "showgrid": False,
                "zeroline": False,
                "showline": True,
                "linecolor": 'black',
                "linewidth": 1,
                "ticks": "outside",
                "tickwidth": 1

            },
            "yaxis": {
                "title_font": {"size": PT6},
                "tickfont": {"size": PT5},
                "color": "black",
                "title_standoff": 5,
                "showgrid": False,
                "zeroline": False,
                "showline": True,
                "linecolor": 'black',
                "linewidth": 1,
                "ticks": "outside",
                "tickwidth": 1

            },
            "legend": {
                "font": {"size": PT6},
                "title": {"font": {"size": PT6}},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0

            },
            "legend2": {
                "font": {"size": PT6},
                "title": {"font": {"size": PT6}},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0
                

            },
            "paper_bgcolor": "rgba(0,0,0,0)",

            "font": {"family": "Arial", "size": PT6, "color": "black"},
            "margin": {"b": 50, "t": 10, "l":30 , "r": 30   },
            "annotationdefaults": {"font": {"family": "Arial", "size": PT6},},
            "coloraxis": {
                "colorbar": {
                    "title": {"font": {"size": PT6}},
                    "tickfont": {"size": PT5},
                    "outlinecolor": "black",   # <--- color of border
                    "outlinewidth": 1,
                },
                "colorscale": [COLORS["jaxpetrol"], COLORS["jaxgold"]]


            },
        }

    )

register_template()