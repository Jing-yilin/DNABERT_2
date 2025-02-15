import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import pandas as pd
from typing import Dict, List, Tuple
import torch
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go

class ExperimentVisualizer:
    def __init__(self, save_dir: str = "results"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('default')
        sns.set_theme(style="whitegrid")
    
    def plot_transfer_matrix(self, transfer_results: List[Dict]) -> None:
        """Plot transfer performance matrix between species pairs."""
        # Extract unique species
        species = set()
        for result in transfer_results:
            species.add(result['source'])
            species.add(result['target'])
        species = sorted(list(species))
        
        # Create performance matrix
        matrix = np.zeros((len(species), len(species)))
        for result in transfer_results:
            i = species.index(result['source'])
            j = species.index(result['target'])
            matrix[i, j] = result['performance']
        
        # Plot heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(matrix, annot=True, fmt='.3f', 
                   xticklabels=species, yticklabels=species,
                   cmap='YlOrRd')
        plt.title('Transfer Performance Matrix')
        plt.xlabel('Target Species')
        plt.ylabel('Source Species')
        plt.tight_layout()
        plt.savefig(self.save_dir / 'transfer_matrix.png')
        plt.close()
    
    def plot_learning_dynamics(self, learning_curves: Dict[str, List[float]]) -> None:
        """Plot detailed learning dynamics for each transfer pair."""
        # Create figure
        plt.figure(figsize=(15, 8))
        
        # Plot each learning curve
        for pair, losses in learning_curves.items():
            epochs = range(1, len(losses) + 1)
            plt.plot(epochs, losses, marker='o', label=pair)
        
        plt.title('Learning Dynamics Across Species Pairs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(self.save_dir / 'learning_dynamics.png')
        plt.close()
        
        # Plot learning rate convergence
        plt.figure(figsize=(12, 6))
        for pair, losses in learning_curves.items():
            epochs = range(1, len(losses) + 1)
            convergence_rate = [abs(losses[i] - losses[i-1]) for i in range(1, len(losses))]
            plt.plot(epochs[1:], convergence_rate, marker='o', label=pair)
        
        plt.title('Convergence Rate Analysis')
        plt.xlabel('Epoch')
        plt.ylabel('Loss Change')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(self.save_dir / 'convergence_analysis.png')
        plt.close()
    
    def plot_feature_space(
        self,
        embeddings: Dict[str, torch.Tensor],
        method: str = 'tsne'
    ) -> None:
        """Visualize feature space distribution using t-SNE or PCA."""
        # Combine embeddings
        all_embeddings = []
        labels = []
        for species, emb in embeddings.items():
            all_embeddings.append(emb.cpu().numpy())
            labels.extend([species] * len(emb))
        
        all_embeddings = np.concatenate(all_embeddings, axis=0)
        
        # Dimensionality reduction
        if method.lower() == 'tsne':
            reducer = TSNE(n_components=2, random_state=42)
        else:
            reducer = PCA(n_components=2, random_state=42)
        
        reduced_embeddings = reducer.fit_transform(all_embeddings)
        
        # Create interactive plot using plotly
        df = pd.DataFrame({
            'x': reduced_embeddings[:, 0],
            'y': reduced_embeddings[:, 1],
            'Species': labels
        })
        
        fig = px.scatter(df, x='x', y='y', color='Species',
                        title=f'Feature Space Distribution ({method.upper()})')
        fig.write_html(str(self.save_dir / f'feature_space_{method}.html'))
        
        # Also save static version
        plt.figure(figsize=(12, 8))
        sns.scatterplot(data=df, x='x', y='y', hue='Species')
        plt.title(f'Feature Space Distribution ({method.upper()})')
        plt.tight_layout()
        plt.savefig(self.save_dir / f'feature_space_{method}.png')
        plt.close()
    
    def plot_attention_patterns(
        self,
        attention_weights: Dict[str, np.ndarray],
        sequence_ids: List[str]
    ) -> None:
        """Visualize attention patterns for different species."""
        for species, weights in attention_weights.items():
            plt.figure(figsize=(15, 10))
            sns.heatmap(weights, xticklabels=sequence_ids, yticklabels=sequence_ids,
                       cmap='viridis')
            plt.title(f'Attention Patterns - {species}')
            plt.tight_layout()
            plt.savefig(self.save_dir / f'attention_patterns_{species}.png')
            plt.close()
    
    def plot_performance_metrics(self, metrics: Dict[str, Dict[str, float]]) -> None:
        """Plot various performance metrics."""
        # Prepare data
        species = list(metrics.keys())
        metric_types = list(metrics[species[0]].keys())
        
        # Create subplots for each metric
        fig, axes = plt.subplots(len(metric_types), 1, figsize=(12, 4*len(metric_types)))
        if len(metric_types) == 1:
            axes = [axes]
        
        for i, metric in enumerate(metric_types):
            values = [metrics[s][metric] for s in species]
            sns.barplot(x=species, y=values, ax=axes[i])
            axes[i].set_title(f'{metric} Across Species')
            axes[i].set_xticklabels(species, rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'performance_metrics.png')
        plt.close()
    
    def create_summary_dashboard(self, results: Dict) -> None:
        """Create an interactive dashboard summarizing all results."""
        # Create a plotly figure with subplots
        fig = go.Figure()
        
        # Add transfer performance matrix
        matrix_data = []
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
        
        heatmap = go.Heatmap(
            z=matrix,
            x=species,
            y=species,
            colorscale='YlOrRd',
            name='Transfer Performance'
        )
        fig.add_trace(heatmap)
        
        # Add learning curves
        for pair, losses in results['learning_curves'].items():
            fig.add_trace(go.Scatter(
                x=list(range(1, len(losses) + 1)),
                y=losses,
                name=f'Learning Curve - {pair}',
                visible=False
            ))
        
        # Create buttons for switching between views
        fig.update_layout(
            updatemenus=[{
                'buttons': [
                    {'label': 'Transfer Matrix',
                     'method': 'update',
                     'args': [{'visible': [True] + [False]*len(results['learning_curves'])}]},
                    {'label': 'Learning Curves',
                     'method': 'update',
                     'args': [{'visible': [False] + [True]*len(results['learning_curves'])}]}
                ],
                'direction': 'down',
                'showactive': True,
            }],
            title='Experiment Results Dashboard'
        )
        
        fig.write_html(str(self.save_dir / 'dashboard.html'))
    
    def plot_evolutionary_distance_impact(
        self,
        transfer_results: List[Dict],
        distance_matrix: Dict[str, Dict[str, float]]
    ) -> None:
        """Analyze the relationship between evolutionary distance and transfer performance."""
        distances = []
        performances = []
        pairs = []
        
        for result in transfer_results:
            source = result['source']
            target = result['target']
            distance = distance_matrix[source][target]
            performance = result['performance']
            
            distances.append(distance)
            performances.append(performance)
            pairs.append(f"{source}->{target}")
        
        # Create scatter plot
        plt.figure(figsize=(12, 8))
        plt.scatter(distances, performances, alpha=0.6)
        
        # Add trend line
        z = np.polyfit(distances, performances, 1)
        p = np.poly1d(z)
        plt.plot(distances, p(distances), "r--", alpha=0.8)
        
        # Add labels for each point
        for i, pair in enumerate(pairs):
            plt.annotate(pair, (distances[i], performances[i]))
        
        plt.xlabel('Evolutionary Distance')
        plt.ylabel('Transfer Performance')
        plt.title('Impact of Evolutionary Distance on Transfer Performance')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(self.save_dir / 'distance_performance_correlation.png')
        plt.close()