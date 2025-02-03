from .search_base import SearchSecondConnectNode
import random

class SearchNodesByRandom(SearchSecondConnectNode):
    def __init__(self, the_method_search_influential_nodes, influential_numbers_by_type,
                 two_hop_neighbors_by_type, primary_type, hg, target_class, clean_labels):
        super().__init__(the_method_search_influential_nodes, influential_numbers_by_type,
                         two_hop_neighbors_by_type, primary_type, hg, target_class, clean_labels)
        self.candidate_nodes_by_type = {}
        self.initial_pool_by_type = {}  # 存储每种类型固定的 2n 个候选节点
        self.prepare_candidate_nodes()

    def prepare_candidate_nodes(self):
        # 预处理候选节点并为每种类型选择固定的 2n 个节点
        for node_type in self.influential_numbers_by_type.keys():
            if node_type in self.two_hop_neighbors_by_type:
                if node_type == self.primary_type:
                    candidate_nodes = {
                        node for node in self.two_hop_neighbors_by_type[node_type]
                        if self.clean_labels[node] == self.target_class
                    }
                else:
                    candidate_nodes = self.two_hop_neighbors_by_type[node_type]
            else:
                print(f"Warning: Node type '{node_type}' does not exist in two_hop_neighbors_by_type.")
                candidate_nodes = set()

            self.candidate_nodes_by_type[node_type] = list(candidate_nodes)

            # 为每种类型选择固定的 2n 个初始节点
            n = self.influential_numbers_by_type[node_type]
            num_initial_nodes = min(2 * n, len(candidate_nodes))
            if candidate_nodes:
                initial_pool = random.sample(list(candidate_nodes), num_initial_nodes)
                self.initial_pool_by_type[node_type] = initial_pool

    def getFilterNodes(self):
        selected_nodes_by_type = {}
        for node_type in self.influential_numbers_by_type.keys():
            n = self.influential_numbers_by_type[node_type]
            initial_pool = self.initial_pool_by_type.get(node_type, [])

            if not initial_pool:
                selected_nodes_by_type[node_type] = []
                continue

            # 从固定的候选池中选择 n 个节点
            selected_nodes = random.sample(initial_pool, min(n, len(initial_pool)))
            selected_nodes_by_type[node_type] = selected_nodes

        return selected_nodes_by_type