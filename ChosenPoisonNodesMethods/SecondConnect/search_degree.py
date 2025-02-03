from .search_base import SearchSecondConnectNode
from utils import get_total_degree
import random

class SearchNodesByDegree(SearchSecondConnectNode):
    def __init__(self, the_method_search_influential_nodes, influential_numbers_by_type, two_hop_neighbors_by_type, primary_type, hg, target_class, clean_labels):
        super().__init__(the_method_search_influential_nodes, influential_numbers_by_type, two_hop_neighbors_by_type, primary_type, hg, target_class, clean_labels)
        self.candidate_nodes_by_type = {}
        self.node_degrees_by_type = {}
        self.prepare_candidate_nodes_and_degrees()

    def prepare_candidate_nodes_and_degrees(self):
        # Precompute candidate nodes and their degrees to avoid repeated calculations
        for node_type in self.influential_numbers_by_type.keys():
            # Get candidate nodes
            if node_type in self.two_hop_neighbors_by_type:
                if node_type == self.primary_type:
                    # Filter nodes with the target class label
                    candidate_nodes = {
                        node for node in self.two_hop_neighbors_by_type[node_type]
                        if self.clean_labels[node] == self.target_class
                    }
                else:
                    candidate_nodes = self.two_hop_neighbors_by_type[node_type]
            else:
                print(f"Warning: Node type '{node_type}' does not exist in two_hop_neighbors_by_type.")
                candidate_nodes = set()
            self.candidate_nodes_by_type[node_type] = candidate_nodes

            # Precompute node degrees
            node_degrees = []
            for node_id in candidate_nodes:
                total_deg = get_total_degree(self.hg, node_id, node_type)
                node_degrees.append((node_id, total_deg))
            self.node_degrees_by_type[node_type] = node_degrees

    def getFilterNodes(self):
        selected_nodes_by_type = {}
        for node_type in self.influential_numbers_by_type.keys():
            # Number of nodes to select
            n = self.influential_numbers_by_type[node_type]
            candidate_nodes = self.candidate_nodes_by_type.get(node_type, [])
            node_degrees = self.node_degrees_by_type.get(node_type, [])
            # Check if candidate nodes are sufficient
            if len(candidate_nodes) < n:
                print(f"Warning: The number of candidate nodes for node type '{node_type}' is insufficient, with only {len(candidate_nodes)} nodes available.")
                n = len(candidate_nodes)
            if not candidate_nodes:
                selected_nodes_by_type[node_type] = []
                continue
            # Sort nodes by degree
            sorted_nodes = sorted(node_degrees, key=lambda x: x[1], reverse=True)
            # Select top n nodes
            top_n_nodes = [node_id for node_id, deg in sorted_nodes[: 2 * n]]
            selected_nodes_by_type[node_type] = random.sample(top_n_nodes, n)
        return selected_nodes_by_type
