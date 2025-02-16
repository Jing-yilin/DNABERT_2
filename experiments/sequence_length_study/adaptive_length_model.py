import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import numpy as np
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import time

class AdaptiveLengthDNABERT2:
    def __init__(
        self,
        base_model_name: str = "bert-base-uncased",
        max_segment_length: int = 512,
        overlap_ratio: float = 0.2
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(base_model_name)
        self.max_segment_length = max_segment_length
        self.overlap_ratio = overlap_ratio
        self.device = torch.device("cpu")
        self.model.to(self.device)
        
    def segment_sequence(self, sequence: str) -> List[str]:
        if len(sequence) <= self.max_segment_length:
            return [sequence]
            
        overlap_size = int(self.max_segment_length * self.overlap_ratio)
        step_size = self.max_segment_length - overlap_size
        
        segments = []
        for i in range(0, len(sequence), step_size):
            segment = sequence[i:i + self.max_segment_length]
            if len(segment) >= self.max_segment_length // 2:  # Only keep segments of reasonable length
                segments.append(segment)
        
        if not segments:  # Fallback for very short sequences
            segments = [sequence]
            
        return segments
    
    def process_segments(self, segments: List[str]) -> torch.Tensor:
        all_embeddings = []
        
        for segment in segments:
            inputs = self.tokenizer(segment, return_tensors='pt')
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state[0]
                
                # Apply attention pooling
                attention_weights = torch.softmax(torch.matmul(embeddings, embeddings.transpose(0, 1)), dim=1)
                segment_embedding = torch.matmul(attention_weights, embeddings)
                segment_embedding = torch.mean(segment_embedding, dim=0)
                
                all_embeddings.append(segment_embedding)
        
        # Combine segment embeddings with attention
        stacked_embeddings = torch.stack(all_embeddings)
        attention_weights = torch.softmax(torch.matmul(stacked_embeddings, stacked_embeddings.transpose(0, 1)), dim=1)
        final_embedding = torch.matmul(attention_weights, stacked_embeddings)
        final_embedding = torch.mean(final_embedding, dim=0)
        
        return final_embedding
    
    def encode_sequence(self, sequence: str) -> torch.Tensor:
        segments = self.segment_sequence(sequence)
        return self.process_segments(segments)
    
    def analyze_sequence_length_impact(
        self,
        sequences: List[str],
        labels: Optional[List[int]] = None,
        save_dir: str = "results"
    ) -> dict:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving results to {save_dir.absolute()}")
        
        results = {
            "sequence_lengths": [],
            "processing_times": [],
            "embedding_similarities": [],
            "attention_patterns": []
        }
        
        for i, seq in enumerate(sequences):
            # Record sequence length
            results["sequence_lengths"].append(len(seq))
            
            # Measure processing time
            start_time = time.time()
            embedding = self.encode_sequence(seq)
            end_time = time.time()
            results["processing_times"].append((end_time - start_time) * 1000)  # Convert to milliseconds
            
            # Store embeddings for similarity analysis
            if i > 0:
                similarity = torch.cosine_similarity(embedding.unsqueeze(0), 
                                                  prev_embedding.unsqueeze(0))
                results["embedding_similarities"].append(similarity.item())
            
            prev_embedding = embedding
        
        # Generate visualizations
        self._plot_length_vs_time(results, save_dir)
        self._plot_length_distribution(results, save_dir)
        if labels is not None:
            self._plot_length_vs_accuracy(results, labels, save_dir)
        
        return results
    
    def _plot_length_vs_time(self, results: dict, save_dir: Path):
        plt.figure(figsize=(10, 6))
        plt.scatter(results["sequence_lengths"], results["processing_times"])
        plt.xlabel("Sequence Length")
        plt.ylabel("Processing Time (ms)")
        plt.title("Sequence Length vs Processing Time")
        plt.savefig(str(save_dir / "length_vs_time.png"))
        plt.close()
    
    def _plot_length_distribution(self, results: dict, save_dir: Path):
        plt.figure(figsize=(10, 6))
        sns.histplot(results["sequence_lengths"], bins=30)
        plt.xlabel("Sequence Length")
        plt.ylabel("Count")
        plt.title("Distribution of Sequence Lengths")
        plt.savefig(str(save_dir / "length_distribution.png"))
        plt.close()
    
    def _plot_length_vs_accuracy(self, results: dict, labels: List[int], save_dir: Path):
        # Group sequences by length ranges and calculate accuracy
        length_ranges = np.linspace(min(results["sequence_lengths"]), 
                                  max(results["sequence_lengths"]), 10)
        accuracies = []
        
        for i in range(len(length_ranges)-1):
            mask = ((np.array(results["sequence_lengths"]) >= length_ranges[i]) & 
                   (np.array(results["sequence_lengths"]) < length_ranges[i+1]))
            if any(mask):
                pred_labels = np.array(labels)[mask]
                accuracy = np.mean(pred_labels == np.array(labels)[mask])
                accuracies.append(accuracy)
            else:
                accuracies.append(0)
        
        plt.figure(figsize=(10, 6))
        plt.plot(length_ranges[:-1], accuracies)
        plt.xlabel("Sequence Length Range")
        plt.ylabel("Accuracy")
        plt.title("Sequence Length vs Accuracy")
        plt.savefig(str(save_dir / "length_vs_accuracy.png"))
        plt.close()