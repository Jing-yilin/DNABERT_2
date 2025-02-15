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

def plot_transfer_results(results: Dict, save_dir: Path):
    """Plot transfer learning results."""
    # Plot 1: Species Distance vs Performance
    plt.figure(figsize=(10, 6))
    distances = [r['species_distance'] for r in results['transfer_performance']]
    performances = [r['performance'] for r in results['transfer_performance']]
    plt.scatter(distances, performances)
    plt.xlabel('Species Distance')
    plt.ylabel('Transfer Performance')
    plt.title('Species Distance vs Transfer Performance')
    plt.savefig(save_dir / 'distance_vs_performance.png')
    plt.close()
    
    # Plot 2: Learning Curves
    plt.figure(figsize=(12, 6))
    for species_pair, history in results['learning_curves'].items():
        plt.plot(history, label=species_pair)
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.title('Transfer Learning Curves')
    plt.legend()
    plt.savefig(save_dir / 'learning_curves.png')
    plt.close()
    
    # Plot 3: Species Distance Matrix Heatmap
    plt.figure(figsize=(10, 8))
    species_list = list(results['distance_matrix'].keys())
    distance_data = np.zeros((len(species_list), len(species_list)))
    for i, s1 in enumerate(species_list):
        for j, s2 in enumerate(species_list):
            distance_data[i, j] = results['distance_matrix'][s1][s2]
    
    sns.heatmap(distance_data, xticklabels=species_list, yticklabels=species_list)
    plt.title('Species Distance Matrix')
    plt.savefig(save_dir / 'distance_matrix.png')
    plt.close()

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