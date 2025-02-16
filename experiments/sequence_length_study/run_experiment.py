import torch
import numpy as np
from adaptive_length_model import AdaptiveLengthDNABERT2
from pathlib import Path
import json
import random
from typing import List, Tuple
import pandas as pd

def generate_synthetic_data(
    num_sequences: int = 100,
    min_length: int = 100,
    max_length: int = 2000
) -> Tuple[List[str], List[int]]:
    nucleotides = ['A', 'C', 'G', 'T']
    sequences = []
    labels = []
    
    for _ in range(num_sequences):
        length = random.randint(min_length, max_length)
        sequence = ''.join(random.choices(nucleotides, k=length))
        
        # Generate synthetic label based on sequence properties
        gc_content = (sequence.count('G') + sequence.count('C')) / length
        label = 1 if gc_content > 0.5 else 0
        
        sequences.append(sequence)
        labels.append(label)
    
    return sequences, labels

def load_real_data(data_path: str) -> Tuple[List[str], List[int]]:
    df = pd.read_csv(data_path)
    return df['sequence'].tolist(), df['label'].tolist()

def run_experiment(
    model: AdaptiveLengthDNABERT2,
    sequences: List[str],
    labels: List[int],
    save_dir: Path
):
    # Analyze sequence length impact
    results = model.analyze_sequence_length_impact(sequences, labels, save_dir)
    
    # Calculate additional metrics
    avg_processing_time = np.mean(results['processing_times'])
    std_processing_time = np.std(results['processing_times'])
    
    # Save metrics
    metrics = {
        'average_processing_time': float(avg_processing_time),
        'std_processing_time': float(std_processing_time),
        'total_sequences': len(sequences),
        'max_sequence_length': max(results['sequence_lengths']),
        'min_sequence_length': min(results['sequence_lengths']),
        'embedding_similarity_mean': float(np.mean(results['embedding_similarities']))
    }
    
    with open(save_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    
    return results, metrics

def main():
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Create save directory
    save_dir = Path('results')
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize model
    model = AdaptiveLengthDNABERT2(
        max_segment_length=512,
        overlap_ratio=0.2
    )
    
    # Generate synthetic data
    print("Generating synthetic data...")
    sequences, labels = generate_synthetic_data(num_sequences=100)
    
    # Run experiment
    print("Running experiment...")
    results, metrics = run_experiment(model, sequences, labels, save_dir)
    
    print("\nExperiment completed!")
    print("\nMetrics:")
    for key, value in metrics.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()