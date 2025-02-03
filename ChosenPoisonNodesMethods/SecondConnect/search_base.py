import random
from abc import ABC, abstractmethod

class SearchSecondConnectNode(ABC):
    def __init__(self, the_method_search_influential_nodes, influential_numbers_by_type,
                 two_hop_neighbors_by_type, primary_type, hg, target_class, clean_labels):
        self.the_method_search_influential_nodes = the_method_search_influential_nodes
        self.influential_numbers_by_type = influential_numbers_by_type
        self.two_hop_neighbors_by_type = two_hop_neighbors_by_type
        self.primary_type = primary_type
        self.hg = hg
        self.target_class = target_class
        self.clean_labels = clean_labels

    @abstractmethod
    def getFilterNodes(self, target_node=None):
        pass

