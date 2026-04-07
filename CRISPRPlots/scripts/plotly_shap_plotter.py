"""
Plotly-based SHAP value visualizations
Converts shap._explanation.Explanation objects to interactive Plotly plots
"""

import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Optional, List, Union
from scipy.stats import gaussian_kde

class PlotlySHAPVisualizer:
    """Convert SHAP Explanation objects to Plotly visualizations"""
    
    def __init__(self, shap_explanation, class_idx=None):
        """
        Initialize with a SHAP Explanation object
        
        Args:
            shap_explanation: shap._explanation.Explanation object
            class_idx: If 3D array, which class to extract (default: 1 for binary classification)
        """
        self.exp = shap_explanation
        values = np.asarray(shap_explanation.values)
        
        # Handle 3D case: extract specific class
        if len(values.shape) == 3:
            if class_idx is None:
                class_idx = 1  # Default to class 1 (negative/second class)
            print(f"Extracting class {class_idx} from 3D array {values.shape}...")
            self.values = values[:, :, class_idx]
        else:
            self.values = values
            
        self.base_value = shap_explanation.base_values
        
        # Convert feature_names to proper list
        if shap_explanation.feature_names is not None:
            fn = shap_explanation.feature_names
            if hasattr(fn, 'tolist'):  # numpy array
                self.feature_names = fn.tolist()
            elif isinstance(fn, np.ndarray):
                self.feature_names = [str(x) for x in fn]
            else:
                self.feature_names = [str(x) for x in fn]
        else:
            self.feature_names = [f"Feature {i}" for i in range(self.values.shape[1])]
        
        self.data = shap_explanation.data
        
        
    def get_df(self):
        mean_abs_shap = np.abs(self.values).mean(axis=0)
        mean_abs_shap = np.asarray(mean_abs_shap).flatten()
        data ={
            "Feature": self.feature_names,
            "MeanAbsSHAP": mean_abs_shap,
        }
        df = pd.DataFrame(data)
        return df
        
    def summary_plot(self, max_features: int = 20, plot_type: str = "dot", colorscale = None, use_features = None, jitter_bin_width=0.01, max_jitter=0.3) -> go.Figure:
        """
        Create a summary plot (bar or dot)
        
        Args:
            max_features: Max features to show zero for all
            plot_type: "bar" or "dot"
        """
        # Calculate mean absolute SHAP values per feature
        mean_abs_shap = np.abs(self.values).mean(axis=0)
        mean_abs_shap = np.asarray(mean_abs_shap).flatten()  # Ensure 1D
        
        # Sort by importance
        
        indices = np.argsort(mean_abs_shap)[::-1]
        # Convert to Python list immediately
        if max_features:
            indices = indices[:max_features]
        indices = indices[::-1].tolist()
        
        sorted_features = [self.feature_names[i] for i in indices]
        sorted_values = [float(mean_abs_shap[i]) for i in indices]
        
        if use_features is not None:
            new_indices = []
            new_features = []
            new_values = []
            for i in range(len(indices)):
                if sorted_features[i] in use_features:
                    new_indices.append(indices[i])
                    new_features.append(sorted_features[i])
                    new_values.append(sorted_values[i])
            sorted_features = new_features
            indices = new_indices
            sorted_values = new_values
            
        
        if plot_type == "bar":
            fig = go.Figure(go.Bar(
                x=sorted_values,
                y=sorted_features,
                orientation='h',
                marker=dict(color=sorted_values, colorscale='Viridis')
            ))
            fig.update_layout(
                title="SHAP Summary Plot (Mean |SHAP value|)",
                xaxis_title="Mean |SHAP value|",
                height=400 + len(sorted_features) * 15,
                showlegend=False
            )
        else:  # dot plot
            fig = self._summary_dot_plot(indices, max_features, colorscale=colorscale, kde_bw=jitter_bin_width, max_jitter=max_jitter)
        
        return fig
    
    def _summary_dot_plot(self, indices: list, max_features: int, colorscale=None, max_jitter=0.3, kde_bw=0.1) -> go.Figure:
        """Beeswarm-style SHAP plot with violin-shaped vertical jitter, handles zero/constant SHAP values"""
        import numpy as np
        import plotly.graph_objects as go

        fig = go.Figure()

        for i, feat_idx in enumerate(indices):
            feat_idx = int(feat_idx)
            shap_vals = self.values[:, feat_idx]
            feature_vals = self.data[:, feat_idx] if self.data is not None else np.arange(len(shap_vals))

            # Normalize feature values for color
            if hasattr(feature_vals, 'shape') and len(feature_vals.shape) > 1:
                feature_vals = feature_vals.flatten()
            try:
                # Compute color limits robustly
                vmin = np.nanpercentile(feature_vals, 5)
                vmax = np.nanpercentile(feature_vals, 95)
                if vmin == vmax:
                    vmin = np.nanpercentile(feature_vals, 1)
                    vmax = np.nanpercentile(feature_vals, 99)
                    if vmin == vmax:
                        vmin = np.min(feature_vals)
                        vmax = np.max(feature_vals)
                if vmin > vmax:  # fix rare numerical precision issues
                    vmin = vmax

                # Normalize feature values to [0,1] for color mapping
                feature_vals_norm = (feature_vals - vmin) / (vmax - vmin + 1e-8)
                feature_vals_norm = np.clip(feature_vals_norm, 0, 1) 
            except:
                feature_vals_norm = np.zeros_like(shap_vals)
            base_y = np.full_like(shap_vals, fill_value=i, dtype=float)

            # Compute density-based jitter safely
            if np.all(shap_vals == shap_vals[0]):
                # All values identical → uniform small jitter
                y_offsets = (np.random.rand(len(shap_vals)) - 0.5) * 2 * max_jitter * 0.1
            else:
                kde = gaussian_kde(shap_vals, bw_method=kde_bw)
                densities = kde(shap_vals)
                densities = densities / densities.max()  # normalize to [0,1]
                # Random vertical offsets proportional to density
                y_offsets = (np.random.rand(len(shap_vals)) - 0.5) * 2 * densities * max_jitter

            y_final = base_y + y_offsets

            fig.add_trace(go.Scatter(
                x=shap_vals,
                y=y_final,
                mode='markers',
                marker=dict(
                    size=6,
                    color=feature_vals_norm,
                    colorscale='Viridis' if colorscale is None else colorscale,
                    showscale=(i == 0),
                    colorbar=dict(title="Norm. value", thickness=10) if i == 0 else None,
                    line=dict(width=0),
                    coloraxis="coloraxis"
                ),
                name=self.feature_names[feat_idx],
                showlegend=False,
            hovertemplate=f"<b>{self.feature_names[feat_idx]}</b><br>SHAP: %{{x:.3f}}<br>Value: %{{marker.color:.3f}}<extra></extra>"            ))

        fig.update_yaxes(
            tickmode="array",
            tickvals=list(range(len(indices))),
            ticktext=[self.feature_names[i] for i in indices],
            range=[-1, len(indices)]
        )


        fig.update_layout(
            xaxis_title="SHAP value",
            hovermode='closest'
        )

        return fig
    
    def force_plot_single(self, sample_idx: int = 0, show_data: bool = True) -> go.Figure:
        """
        Create force plot for a single sample (horizontal waterfall)
        
        Args:
            sample_idx: Which sample to visualize
            show_data: Include original feature values
        """
        shap_vals = self.values[sample_idx]
        indices = np.argsort(np.abs(shap_vals))[::-1][:15].tolist()  # Convert to list immediately
        
        sorted_features = [self.feature_names[i] for i in indices]
        sorted_shap = [float(shap_vals[i]) for i in indices]
        
        # Calculate cumulative values
        cumsum = np.concatenate([[self.base_value], np.cumsum(sorted_shap[:-1])])
        
        fig = go.Figure(go.Waterfall(
            x=sorted_shap,
            y=sorted_features + ["Output"],
            measure=["relative"] * len(sorted_features) + ["total"],
            connector=dict(line=dict(color="rgba(0,0,0,0.5)")),
            increasing=dict(marker=dict(color="rgba(31, 119, 180, 0.8)")),
            decreasing=dict(marker=dict(color="rgba(255, 7, 58, 0.8)"))
        ))
        
        fig.update_layout(
            title=f"SHAP Force Plot - Sample {sample_idx}",
            xaxis_title="SHAP value contribution",
            height=400 + len(sorted_features) * 20,
            showlegend=False
        )
        
        return fig
    
    def dependence_plot(self, feature_idx: Union[int, str], interaction_idx: Optional[Union[int, str]] = None) -> go.Figure:
        """
        Create dependence plot (scatter of feature vs SHAP value)
        
        Args:
            feature_idx: Feature name or index
            interaction_idx: Optional second feature for color coding
        """
        if isinstance(feature_idx, str):
            feature_idx = self.feature_names.index(feature_idx)
        
        feature_idx = int(feature_idx)
        
        if self.data is None:
            return go.Figure().add_annotation(text="No data available for dependence plot")
        
        x_vals = self.data[:, feature_idx]
        y_vals = self.values[:, feature_idx]
        
        if interaction_idx is not None:
            if isinstance(interaction_idx, str):
                interaction_idx = self.feature_names.index(interaction_idx)
            interaction_idx = int(interaction_idx)
            color_vals = self.data[:, interaction_idx]
            color_name = self.feature_names[interaction_idx]
        else:
            color_vals = y_vals
            color_name = "SHAP value"
        
        fig = go.Figure(go.Scatter(
            x=x_vals,
            y=y_vals,
            mode='markers',
            marker=dict(
                size=8,
                color=color_vals,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title=color_name),
                line=dict(width=0)
            ),
            hovertemplate=f"<b>{self.feature_names[feature_idx]}</b>: %{{x:.3f}}<br>" +
                         f"<b>SHAP</b>: %{{y:.3f}}<extra></extra>"
        ))
        
        fig.update_layout(
            title=f"Dependence Plot: {self.feature_names[feature_idx]}",
            xaxis_title=self.feature_names[feature_idx],
            yaxis_title="SHAP value",
            height=500,
            width=700
        )
        
        return fig
    
    def bar_plot(self, max_features: int = 15) -> go.Figure:
        """Mean SHAP values (signed) for each feature"""
        mean_shap = self.values.mean(axis=0)
        mean_shap = np.asarray(mean_shap).flatten()
        indices = np.argsort(np.abs(mean_shap))[::-1][:max_features].tolist()
        
        sorted_features = [self.feature_names[i] for i in indices]
        sorted_values = [float(mean_shap[i]) for i in indices]
        colors = ['rgba(31, 119, 180, 0.8)' if x > 0 else 'rgba(255, 7, 58, 0.8)' for x in sorted_values]
        
        fig = go.Figure(go.Bar(
            x=sorted_values,
            y=sorted_features,
            orientation='h',
            marker=dict(color=colors),
            hovertemplate="<b>%{y}</b><br>Mean SHAP: %{x:.4f}<extra></extra>"
        ))
        
        fig.update_layout(
            title="Mean SHAP Values by Feature",
            xaxis_title="Mean SHAP value",
            height=400 + len(sorted_features) * 15,
            showlegend=False
        )
        
        return fig


def load_and_visualize(pickle_path: str, plot_type: str = "summary") -> go.Figure:
    """
    Load a SHAP Explanation pickle and create visualization
    
    Args:
        pickle_path: Path to the pickle file
        plot_type: "summary", "bar", "force", "dependence"
    """
    print(f"Loading {pickle_path}...")
    with open(pickle_path, "rb") as f:
        shap_exp = pickle.load(f)
    
    print(f"Creating {plot_type} plot...")
    viz = PlotlySHAPVisualizer(shap_exp)
    
    if plot_type == "summary":
        fig = viz.summary_plot(plot_type="dot")
    elif plot_type == "bar":
        fig = viz.bar_plot()
    elif plot_type == "force":
        fig = viz.force_plot_single(sample_idx=0)
    elif plot_type == "dependence":
        fig = viz.dependence_plot(feature_idx=0)
    else:
        raise ValueError(f"Unknown plot_type: {plot_type}")
    
    return fig


if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize SHAP Explanation objects with Plotly")
    parser.add_argument("pickle_file", help="Path to SHAP pickle file")
    parser.add_argument("--plot-type", default="summary", 
                       choices=["summary", "bar", "force", "dependence"],
                       help="Type of plot to create")
    parser.add_argument("--output", default=None, help="Output HTML file (optional)")
    parser.add_argument("--show", action="store_true", help="Show in browser")
    
    args = parser.parse_args()
    
    fig = load_and_visualize(args.pickle_file, plot_type=args.plot_type)
    
    if args.output:
        fig.write_html(args.output)
        print(f"✅ Saved to {args.output}")
    
    if args.show:
        fig.show()
    else:
        print("Use --show to display in browser or --output <file> to save")