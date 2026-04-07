import math
import plotly.graph_objects as go
from plotly_template import COLORS

# Values n = 1,...,6
n_vals = list(range(1, 7))

# Compute binomial coefficients
binoms = [math.comb(23, n) for n in n_vals]

# Normalization constant
Z = sum(binoms)

# Conditional probabilities
probs = [b / Z for b in binoms]

# Create bar plot
fig = go.Figure(
    data=go.Bar(
        x=n_vals,
        y=probs,
        marker=dict(color=COLORS["jaxpetrol"], line=dict(color="black", width=1))
        
    )
)

fig.update_layout(
    template="simple_white_custom",
    width=80,
    height=30,
    margin=dict(r=0, l=0, b=0, t=0)
)

fig.write_image("Figures/BarChartDistribution.svg")