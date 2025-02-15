import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import List, Dict, Optional
import numpy as np
from pathlib import Path
import json
import time

class AdaptiveTransferDNABERT:
    def __init__(
        self,
        base_model_name: str = "bert-base-uncased",
        device: str = "cpu",
        species_similarity_matrix: Optional[Dict[str, Dict[str, float]]] = None
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.model = AutoModel.from_pretrained(base_model_name)
        self.device = torch.device(device)
        self.model.to(self.device)
        
        # Initialize or load species similarity matrix
        self.species_similarity_matrix = species_similarity_matrix or {}
        
        # Track adaptation history
        self.adaptation_history = []
    
    def compute_species_distance(self, source_species: str, target_species: str) -> float:
        """Compute evolutionary distance between species."""
        if source_species in self.species_similarity_matrix and \
           target_species in self.species_similarity_matrix[source_species]:
            return self.species_similarity_matrix[source_species][target_species]
        return 1.0  # Maximum distance if not found
    
    def adapt_learning_rate(self, species_distance: float, base_lr: float = 1e-5) -> float:
        """Adjust learning rate based on species distance."""
        # Closer species need less aggressive adaptation
        return base_lr * (1 + species_distance)
    
    def adapt_layers(self, species_distance: float) -> List[bool]:
        """Determine which layers to fine-tune based on species distance."""
        num_layers = self.model.config.num_hidden_layers
        # More distant species require more layers to be fine-tuned
        num_layers_to_tune = int(np.ceil(num_layers * species_distance))
        return [i >= (num_layers - num_layers_to_tune) for i in range(num_layers)]
    
    def prepare_for_transfer(
        self,
        source_species: str,
        target_species: str,
        base_lr: float = 1e-5
    ) -> Dict:
        """Prepare model for transfer learning between species."""
        species_distance = self.compute_species_distance(source_species, target_species)
        lr = self.adapt_learning_rate(species_distance, base_lr)
        trainable_layers = self.adapt_layers(species_distance)
        
        # Freeze/unfreeze layers according to species distance
        for name, param in self.model.named_parameters():
            layer_idx = int(name.split('.')[2]) if 'layer' in name else -1
            if layer_idx >= 0:
                param.requires_grad = trainable_layers[layer_idx]
        
        return {
            'species_distance': species_distance,
            'learning_rate': lr,
            'trainable_layers': trainable_layers
        }
    
    def encode_sequence(self, sequence: str) -> torch.Tensor:
        """Encode a DNA sequence to embeddings."""
        inputs = self.tokenizer(sequence, return_tensors='pt')
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state[0]
            # Use mean pooling
            sequence_embedding = torch.mean(embeddings, dim=0)
        
        return sequence_embedding
    
    def train_step(
        self,
        batch_sequences: List[str],
        batch_labels: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module
    ) -> float:
        """Perform one training step."""
        self.model.train()
        optimizer.zero_grad()
        
        # Process batch
        inputs = self.tokenizer(batch_sequences, padding=True, return_tensors='pt')
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        labels = batch_labels.to(self.device)
        
        # Forward pass
        outputs = self.model(**inputs)
        logits = torch.mean(outputs.last_hidden_state, dim=1)
        
        # Compute loss
        loss = criterion(logits, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        return loss.item()
    
    def evaluate(
        self,
        eval_sequences: List[str],
        eval_labels: torch.Tensor
    ) -> Dict[str, float]:
        """Evaluate model performance."""
        self.model.eval()
        total_loss = 0
        criterion = nn.MSELoss()
        
        with torch.no_grad():
            for i in range(0, len(eval_sequences), 32):
                batch_sequences = eval_sequences[i:i+32]
                batch_labels = eval_labels[i:i+32]
                
                inputs = self.tokenizer(batch_sequences, padding=True, return_tensors='pt')
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                labels = batch_labels.to(self.device)
                
                outputs = self.model(**inputs)
                logits = torch.mean(outputs.last_hidden_state, dim=1)
                loss = criterion(logits, labels)
                total_loss += loss.item()
        
        avg_loss = total_loss / (len(eval_sequences) / 32)
        return {'eval_loss': avg_loss}
    
    def save_adaptation_state(
        self,
        save_dir: str,
        source_species: str,
        target_species: str
    ):
        """Save adaptation state and history."""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        self.model.save_pretrained(save_path / "model")
        self.tokenizer.save_pretrained(save_path / "tokenizer")
        
        # Save adaptation history
        history = {
            'source_species': source_species,
            'target_species': target_species,
            'adaptation_history': self.adaptation_history,
            'species_similarity_matrix': self.species_similarity_matrix
        }
        
        with open(save_path / "adaptation_history.json", 'w') as f:
            json.dump(history, f, indent=2)
    
    def load_adaptation_state(self, load_dir: str):
        """Load adaptation state and history."""
        load_path = Path(load_dir)
        
        # Load model and tokenizer
        self.model = AutoModel.from_pretrained(load_path / "model")
        self.tokenizer = AutoTokenizer.from_pretrained(load_path / "tokenizer")
        
        # Load adaptation history
        with open(load_path / "adaptation_history.json", 'r') as f:
            history = json.load(f)
            self.adaptation_history = history['adaptation_history']
            self.species_similarity_matrix = history['species_similarity_matrix']
        
        self.model.to(self.device)