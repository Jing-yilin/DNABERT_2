import torch
import torch.nn as nn
from models.adaptive_transfer import AdaptiveTransferDNABERT
from utils.species_distance import SpeciesDistanceCalculator
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple
import time

def generate_synthetic_data(species: str, num_sequences: int = 100) -> Tuple[List[str], torch.Tensor]:
    """Generate synthetic DNA sequences and labels for a species."""
    nucleotides = ['A', 'C', 'G', 'T']
    sequences = []
    labels = []
    
    # Add species-specific bias
    if 'sapiens' in species.lower():
        gc_bias = 0.6
    elif 'musculus' in species.lower():
        gc_bias = 0.5
    else:
        gc_bias = 0.4
    
    for _ in range(num_sequences):
        # Generate sequence with species-specific GC content
        sequence = []
        for _ in range(500):  # Fixed length sequences
            if np.random.random() < gc_bias:
                base = np.random.choice(['G', 'C'])
            else:
                base = np.random.choice(['A', 'T'])
            sequence.append(base)
        
        sequence = ''.join(sequence)
        sequences.append(sequence)
        
        # Generate label (e.g., GC content as a continuous value)
        gc_content = (sequence.count('G') + sequence.count('C')) / len(sequence)
        labels.append(gc_content)
    
    return sequences, torch.tensor(labels, dtype=torch.float32)

from utils.visualization import ExperimentVisualizer
from utils.advanced_visualization import AdvancedVisualizer
import time
import psutil
import numpy as np

def plot_transfer_results(results: Dict, save_dir: Path):
    """Plot transfer learning results with enhanced visualizations."""
    # Basic visualizations
    visualizer = ExperimentVisualizer(str(save_dir))
    visualizer.plot_transfer_matrix(results['transfer_performance'])
    visualizer.plot_learning_dynamics(results['learning_curves'])
    visualizer.plot_evolutionary_distance_impact(
        results['transfer_performance'],
        results['distance_matrix']
    )
    
    # Advanced visualizations
    advanced_viz = AdvancedVisualizer(str(save_dir))
    
    # Training dynamics visualization
    advanced_viz.plot_training_dynamics(results['learning_curves'])
    
    # Performance metrics
    performance_metrics = {}
    for result in results['transfer_performance']:
        species_pair = f"{result['source']}->{result['target']}"
        performance_metrics[species_pair] = {
            'transfer_performance': result['performance'],
            'relative_efficiency': result['performance'] / result['species_distance'],
            'convergence_speed': len(results['learning_curves'][species_pair]),
            'stability': np.std(results['learning_curves'][species_pair])
        }
    advanced_viz.plot_performance_comparison(performance_metrics)
    
    # Resource usage tracking
    resource_metrics = {
        'cpu_percent': [],
        'memory_percent': [],
        'memory_used_gb': []
    }
    
    # Simulate resource tracking
    for _ in range(10):
        resource_metrics['cpu_percent'].append(psutil.cpu_percent())
        resource_metrics['memory_percent'].append(psutil.virtual_memory().percent)
        resource_metrics['memory_used_gb'].append(psutil.virtual_memory().used / (1024**3))
        time.sleep(0.1)
    
    advanced_viz.plot_resource_usage(resource_metrics)
    
    # Attention pattern analysis
    attention_data = {}
    for result in results['transfer_performance']:
        source = result['source']
        target = result['target']
        # Simulate attention patterns
        attention_data[f"{source}->{target}"] = np.random.rand(10, 10)
    
    advanced_viz.plot_attention_analysis(attention_data)
    
    # Create interactive dashboard with all metrics
    dashboard_data = {
        **results,
        'performance_metrics': performance_metrics,
        'resource_usage': resource_metrics,
        'attention_patterns': attention_data
    }
    advanced_viz.create_interactive_dashboard(dashboard_data)

def run_transfer_experiment(
    model: AdaptiveTransferDNABERT,
    source_species: str,
    target_species: str,
    num_epochs: int = 5,
    batch_size: int = 32
) -> Dict:
    """Run transfer learning experiment between two species."""
    # Prepare data
    source_sequences, source_labels = generate_synthetic_data(source_species)
    target_sequences, target_labels = generate_synthetic_data(target_species)
    
    # Prepare model for transfer
    transfer_config = model.prepare_for_transfer(source_species, target_species)
    
    # Initialize optimizer and criterion
    optimizer = torch.optim.Adam(
        [p for p in model.model.parameters() if p.requires_grad],
        lr=transfer_config['learning_rate']
    )
    criterion = nn.MSELoss()
    
    # Training loop
    losses = []
    for epoch in range(num_epochs):
        epoch_losses = []
        for i in range(0, len(target_sequences), batch_size):
            batch_sequences = target_sequences[i:i+batch_size]
            batch_labels = target_labels[i:i+batch_size]
            
            loss = model.train_step(batch_sequences, batch_labels, optimizer, criterion)
            epoch_losses.append(loss)
        
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    # Evaluate
    eval_metrics = model.evaluate(target_sequences[-batch_size:], target_labels[-batch_size:])
    
    return {
        'transfer_config': transfer_config,
        'training_losses': losses,
        'eval_metrics': eval_metrics
    }

def main():
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create save directory
    save_dir = Path('experiments/cross_species_transfer/results')
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize species distance calculator
    calculator = SpeciesDistanceCalculator(SpeciesDistanceCalculator.get_example_newick_tree())
    
    # Define species for experiment
    species_list = [
        'Homo_sapiens',
        'Mus_musculus',
        'Danio_rerio',
        'Drosophila_melanogaster'
    ]
    
    # Compute distance matrix
    distance_matrix = calculator.compute_distance_matrix(species_list)
    
    # Initialize model
    model = AdaptiveTransferDNABERT(species_similarity_matrix=distance_matrix)
    
    # Run experiments
    results = {
        'distance_matrix': distance_matrix,
        'transfer_performance': [],
        'learning_curves': {}
    }
    
    # Run transfer learning experiments between species pairs
    for source_species in species_list[:-1]:
        for target_species in species_list[1:]:
            if source_species != target_species:
                print(f"\nRunning transfer from {source_species} to {target_species}")
                experiment_results = run_transfer_experiment(
                    model, source_species, target_species
                )
                
                # Record results
                results['transfer_performance'].append({
                    'source': source_species,
                    'target': target_species,
                    'species_distance': distance_matrix[source_species][target_species],
                    'performance': experiment_results['eval_metrics']['eval_loss']
                })
                
                results['learning_curves'][f"{source_species}->{target_species}"] = \
                    experiment_results['training_losses']
    
    # Plot results
    plot_transfer_results(results, save_dir)
    
    # Save results
    with open(save_dir / 'experiment_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nExperiment completed! Results saved to:", save_dir)

if __name__ == "__main__":
    main()