import abc


class BasePoisonNodeSelector(abc.ABC):
    def __init__(self, homo_g, node_mapping, primary_type):
        self.homo_g = homo_g
        self.node_mapping = node_mapping
        self.primary_type = primary_type

    def _map_victim_nodes(self, victim_nodes):
        return [self.node_mapping[(self.primary_type, node)] for node in victim_nodes]

    @abc.abstractmethod
    def select_poison_nodes(self, victim_nodes, num_poison_nodes):
        pass