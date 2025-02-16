from typing import Dict, List, Tuple
import numpy as np
from Bio import Phylo
from io import StringIO
import json
from pathlib import Path

class SpeciesDistanceCalculator:
    def __init__(self, newick_tree: str = None):
        """Initialize with a phylogenetic tree in Newick format."""
        self.tree = None
        if newick_tree:
            self.tree = Phylo.read(StringIO(newick_tree), 'newick')
    
    def compute_distance_matrix(self, species_list: List[str]) -> Dict[str, Dict[str, float]]:
        """Compute pairwise distances between species."""
        distance_matrix = {}
        
        for species1 in species_list:
            distance_matrix[species1] = {}
            for species2 in species_list:
                if species1 == species2:
                    distance_matrix[species1][species2] = 0.0
                else:
                    distance = self._compute_evolutionary_distance(species1, species2)
                    distance_matrix[species1][species2] = distance
        
        return distance_matrix
    
    def _compute_evolutionary_distance(self, species1: str, species2: str) -> float:
        """Compute evolutionary distance between two species."""
        if self.tree:
            try:
                distance = self.tree.distance(species1, species2)
                # Normalize distance to [0, 1]
                max_distance = max(self.tree.depths().values())
                return distance / max_distance
            except Exception as e:
                print(f"Error computing tree distance: {e}")
                return self._compute_fallback_distance(species1, species2)
        else:
            return self._compute_fallback_distance(species1, species2)
    
    def _compute_fallback_distance(self, species1: str, species2: str) -> float:
        """Compute a fallback distance when tree is not available."""
        # Simple taxonomic distance based on species names
        parts1 = species1.split('_')
        parts2 = species2.split('_')
        
        # Compare genus (first part of species name)
        if parts1[0] == parts2[0]:
            return 0.3  # Same genus
        return 0.7  # Different genus
    
    def normalize_distances(self, distance_matrix: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """Normalize distances to [0, 1] range."""
        all_distances = []
        for species1 in distance_matrix:
            for species2 in distance_matrix[species1]:
                all_distances.append(distance_matrix[species1][species2])
        
        max_dist = max(all_distances)
        min_dist = min(all_distances)
        range_dist = max_dist - min_dist
        
        normalized_matrix = {}
        for species1 in distance_matrix:
            normalized_matrix[species1] = {}
            for species2 in distance_matrix[species1]:
                if range_dist == 0:
                    normalized_matrix[species1][species2] = 0.0
                else:
                    dist = distance_matrix[species1][species2]
                    normalized_matrix[species1][species2] = (dist - min_dist) / range_dist
        
        return normalized_matrix
    
    def save_distance_matrix(self, matrix: Dict[str, Dict[str, float]], file_path: str):
        """Save distance matrix to file."""
        with open(file_path, 'w') as f:
            json.dump(matrix, f, indent=2)
    
    def load_distance_matrix(self, file_path: str) -> Dict[str, Dict[str, float]]:
        """Load distance matrix from file."""
        with open(file_path, 'r') as f:
            return json.load(f)
    
    @staticmethod
    def get_example_newick_tree() -> str:
        """Return an example Newick tree for common model organisms."""
        return """
        (
            (
                (
                    (Homo_sapiens:0.1,Pan_troglodytes:0.1):0.1,
                    Mus_musculus:0.2
                ):0.1,
                (
                    Danio_rerio:0.3,
                    Xenopus_tropicalis:0.3
                ):0.1
            ):0.1,
            (
                Drosophila_melanogaster:0.4,
                Caenorhabditis_elegans:0.4
            ):0.1
        );
        """