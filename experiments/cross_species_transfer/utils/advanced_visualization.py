import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import pandas as pd
from typing import Dict, List, Tuple
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class AdvancedVisualizer:
    def __init__(self, save_dir: str = "results"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('default')
        sns.set_theme(style="whitegrid")
        plt.rcParams['figure.dpi'] = 300
        plt.rcParams['savefig.dpi'] = 300
    
    def plot_training_dynamics(self, learning_curves: Dict[str, List[float]]) -> None:
        """Create detailed training dynamics visualization."""
        # Create figure with secondary y-axis
        fig = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(2, 2)
        
        # Plot 1: Learning curves
        ax1 = fig.add_subplot(gs[0, :])
        for pair, losses in learning_curves.items():
            ax1.plot(range(1, len(losses) + 1), losses, marker='o', label=pair)
        ax1.set_title('Learning Curves Across Species Pairs')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Plot 2: Loss distribution
        ax2 = fig.add_subplot(gs[1, 0])
        all_losses = []
        pairs = []
        for pair, losses in learning_curves.items():
            all_losses.extend(losses)
            pairs.extend([pair] * len(losses))
        sns.violinplot(data=pd.DataFrame({'Loss': all_losses, 'Pair': pairs}),
                      x='Pair', y='Loss', ax=ax2)
        ax2.set_title('Loss Distribution by Species Pair')
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
        
        # Plot 3: Convergence speed
        ax3 = fig.add_subplot(gs[1, 1])
        convergence_epochs = []
        pair_names = []
        for pair, losses in learning_curves.items():
            # Define convergence as loss change < 0.01
            for i in range(1, len(losses)):
                if abs(losses[i] - losses[i-1]) < 0.01:
                    convergence_epochs.append(i)
                    pair_names.append(pair)
                    break
        sns.barplot(x=pair_names, y=convergence_epochs, ax=ax3)
        ax3.set_title('Convergence Speed by Species Pair')
        ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45)
        ax3.set_ylabel('Epochs to Converge')
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'training_dynamics.png', bbox_inches='tight')
        plt.close()
    
    def plot_performance_comparison(self, metrics: Dict[str, Dict[str, float]]) -> None:
        """Create comprehensive performance comparison visualization."""
        # Prepare data
        pairs = list(metrics.keys())
        metric_types = list(metrics[pairs[0]].keys())
        
        # Create figure with subplots
        fig = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(2, 2)
        
        # Plot 1: Overall performance radar chart
        ax1 = fig.add_subplot(gs[0, 0], projection='polar')
        angles = np.linspace(0, 2*np.pi, len(metric_types), endpoint=False)
        angles = np.concatenate((angles, [angles[0]]))  # complete the circle
        
        for pair in pairs:
            values = [metrics[pair][metric] for metric in metric_types]
            values = np.concatenate((values, [values[0]]))
            ax1.plot(angles, values, 'o-', label=pair)
            ax1.fill(angles, values, alpha=0.25)
        ax1.set_xticks(angles[:-1])
        ax1.set_xticklabels(metric_types)
        ax1.set_title('Performance Radar Chart')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Plot 2: Performance heatmap
        ax2 = fig.add_subplot(gs[0, 1])
        heatmap_data = np.zeros((len(pairs), len(metric_types)))
        for i, pair in enumerate(pairs):
            for j, metric in enumerate(metric_types):
                heatmap_data[i, j] = metrics[pair][metric]
        sns.heatmap(heatmap_data, xticklabels=metric_types, yticklabels=pairs,
                   annot=True, fmt='.3f', cmap='YlOrRd', ax=ax2)
        ax2.set_title('Performance Heatmap')
        
        # Plot 3: Performance distribution
        ax3 = fig.add_subplot(gs[1, :])
        data = []
        metric_names = []
        pair_names = []
        for pair in pairs:
            for metric in metric_types:
                data.append(metrics[pair][metric])
                metric_names.append(metric)
                pair_names.append(pair)
        df = pd.DataFrame({
            'Metric': metric_names,
            'Value': data,
            'Pair': pair_names
        })
        sns.boxplot(data=df, x='Metric', y='Value', hue='Pair', ax=ax3)
        ax3.set_title('Performance Distribution')
        ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'performance_comparison.png', bbox_inches='tight')
        plt.close()
    
    def create_interactive_dashboard(self, results: Dict) -> None:
        """Create an interactive HTML dashboard with Plotly."""
        # Create subplot figure
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Transfer Performance Matrix', 'Learning Curves',
                          'Performance Metrics', 'Resource Usage'),
            specs=[[{'type': 'heatmap'}, {'type': 'scatter'}],
                  [{'type': 'bar'}, {'type': 'scatter'}]]
        )
        
        # 1. Transfer Performance Matrix
        species = set()
        for result in results['transfer_performance']:
            species.add(result['source'])
            species.add(result['target'])
        species = sorted(list(species))
        
        matrix = np.zeros((len(species), len(species)))
        for result in results['transfer_performance']:
            i = species.index(result['source'])
            j = species.index(result['target'])
            matrix[i, j] = result['performance']
        
        fig.add_trace(
            go.Heatmap(z=matrix, x=species, y=species,
                      colorscale='YlOrRd', name='Transfer Performance'),
            row=1, col=1
        )
        
        # 2. Learning Curves
        for pair, losses in results['learning_curves'].items():
            fig.add_trace(
                go.Scatter(x=list(range(1, len(losses) + 1)),
                          y=losses, name=f'Learning Curve - {pair}',
                          mode='lines+markers'),
                row=1, col=2
            )
        
        # 3. Performance Metrics
        metrics = {}
        for result in results['transfer_performance']:
            pair = f"{result['source']}->{result['target']}"
            metrics[pair] = result['performance']
        
        fig.add_trace(
            go.Bar(x=list(metrics.keys()),
                  y=list(metrics.values()),
                  name='Performance'),
            row=2, col=1
        )
        
        # 4. Resource Usage
        if 'resource_usage' in results:
            for metric, values in results['resource_usage'].items():
                fig.add_trace(
                    go.Scatter(x=list(range(len(values))),
                             y=values,
                             name=f'Resource - {metric}',
                             mode='lines'),
                    row=2, col=2
                )
        
        # Update layout
        fig.update_layout(
            height=1000,
            width=1500,
            title_text="Cross-Species Transfer Learning Dashboard",
            showlegend=True
        )
        
        # Save dashboard
        fig.write_html(str(self.save_dir / 'interactive_dashboard.html'))
    
    def plot_resource_usage(self, resource_metrics: Dict[str, List[float]]) -> None:
        """Create detailed resource usage visualization."""
        fig = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(2, 2)
        
        # Plot 1: Resource usage over time
        ax1 = fig.add_subplot(gs[0, :])
        for metric, values in resource_metrics.items():
            ax1.plot(range(len(values)), values, label=metric)
        ax1.set_title('Resource Usage Over Time')
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Usage')
        ax1.legend()
        
        # Plot 2: Resource usage distribution
        ax2 = fig.add_subplot(gs[1, 0])
        data = []
        metric_names = []
        for metric, values in resource_metrics.items():
            data.extend(values)
            metric_names.extend([metric] * len(values))
        sns.violinplot(data=pd.DataFrame({'Usage': data, 'Metric': metric_names}),
                      x='Metric', y='Usage', ax=ax2)
        ax2.set_title('Resource Usage Distribution')
        
        # Plot 3: Peak usage comparison
        ax3 = fig.add_subplot(gs[1, 1])
        peak_usage = {metric: max(values) for metric, values in resource_metrics.items()}
        sns.barplot(x=list(peak_usage.keys()), y=list(peak_usage.values()), ax=ax3)
        ax3.set_title('Peak Resource Usage')
        ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'resource_usage.png', bbox_inches='tight')
        plt.close()
    
    def plot_attention_analysis(self, attention_data: Dict[str, np.ndarray]) -> None:
        """Create attention pattern analysis visualization."""
        fig = plt.figure(figsize=(20, 15))
        gs = fig.add_gridspec(2, 2)
        
        # Plot 1: Attention heatmaps
        ax1 = fig.add_subplot(gs[0, 0])
        combined_attention = np.mean([att for att in attention_data.values()], axis=0)
        sns.heatmap(combined_attention, ax=ax1, cmap='viridis')
        ax1.set_title('Average Attention Patterns')
        
        # Plot 2: Attention pattern comparison
        ax2 = fig.add_subplot(gs[0, 1])
        attention_similarities = np.zeros((len(attention_data), len(attention_data)))
        species = list(attention_data.keys())
        for i, s1 in enumerate(species):
            for j, s2 in enumerate(species):
                similarity = np.corrcoef(attention_data[s1].flatten(),
                                      attention_data[s2].flatten())[0, 1]
                attention_similarities[i, j] = similarity
        sns.heatmap(attention_similarities, xticklabels=species, yticklabels=species,
                   ax=ax2, cmap='YlOrRd')
        ax2.set_title('Attention Pattern Similarities')
        
        # Plot 3: Attention focus analysis
        ax3 = fig.add_subplot(gs[1, :])
        focus_scores = []
        species_names = []
        positions = []
        for species, att in attention_data.items():
            for pos in range(att.shape[1]):
                focus_scores.append(np.mean(att[:, pos]))
                species_names.append(species)
                positions.append(pos)
        df = pd.DataFrame({
            'Position': positions,
            'Focus Score': focus_scores,
            'Species': species_names
        })
        sns.lineplot(data=df, x='Position', y='Focus Score', hue='Species', ax=ax3)
        ax3.set_title('Attention Focus by Position')
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'attention_analysis.png', bbox_inches='tight')
        plt.close()