from .base_selector import BasePoisonNodeSelector


class DegreeBasedSelector(BasePoisonNodeSelector):
    def select_poison_nodes(self, victim_nodes, num_poison_nodes):
        mapped_nodes = self._map_victim_nodes(victim_nodes)
        degree_dict = dict(self.homo_g.degree())

        sorted_target_nodes = sorted(mapped_nodes,
                                     key=lambda x: degree_dict[x],
                                     reverse=True)
        return sorted_target_nodes[:num_poison_nodes]