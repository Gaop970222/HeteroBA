import copy
import math
import numpy as np
from utils import get_src_dst_from_etype, get_total_degree
from ChosenPoisonNodesMethods.SecondConnect import create_victim_model
from scipy.stats import gaussian_kde
from tqdm import tqdm
import torch
from .backdoor import Backdoor
#------debug----

class HeteroBA(Backdoor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.the_method_search_influential_nodes = args[0].chosen_influential_nodes_method
        influential_numbers_by_type, two_hop_neighbors_by_type = self.select_influential_nodes()
        self.second_search_method = create_victim_model(self.the_method_search_influential_nodes,
                                                   influential_numbers_by_type,
                                                   two_hop_neighbors_by_type, self.primary_type, self.hg,
                                                   self.target_class,
                                                   self.clean_labels, self.metapaths, self.trainset_index,
                                                   self.testset_index,
                                                   self.args, self.text_attribute, self.edge_template)

    def construct_posion_graph(self, training = True):
        poison_hg = copy.deepcopy(self.hg)
        GREEN_TEXT = "\033[92m"
        RESET = "\033[0m"
        if training:
            size = len(self.posion_trainset_index)
        else:
            size = len(self.posion_testset_index)
        for i in tqdm(range(size), colour='blue', desc=f"{GREEN_TEXT}Constructing Posion Graph....{RESET}"):
            if training:
                poison_node_id = self.posion_trainset_index[i]
            else:
                poison_node_id = self.posion_testset_index[i]
            new_trigger_node_id = max(poison_hg.nodes(self.trigger_type))+1
            new_feature = self.calculate_features(new_trigger_node_id)
            poison_hg.add_nodes(1, ntype=self.trigger_type,data = {'inp':new_feature} )
            meta_relation = self.trigger_type[0] + self.primary_type[0]
            reversed_meta_relation = self.primary_type[0] + self.trigger_type[0]
            poison_hg.add_edges(new_trigger_node_id, poison_node_id, etype=meta_relation)
            poison_hg.add_edges(poison_node_id,new_trigger_node_id,  etype=reversed_meta_relation)
            if  'attention' in self.the_method_search_influential_nodes:
                filter_nodes = self.second_search_method.getFilterNodes(target_node = poison_node_id)
            else:
                filter_nodes = self.second_search_method.getFilterNodes()
            print("filter nodes are :", filter_nodes)
            for k in filter_nodes.keys():
                nodes = filter_nodes[k]
                relation = self.trigger_type[0] + k[0]
                reverse_relation = k[0] + self.trigger_type[0]
                for node in nodes:
                    poison_hg.add_edges(new_trigger_node_id, node, etype=relation)
                    poison_hg.add_edges(node, new_trigger_node_id, etype=reverse_relation)

        return poison_hg

    def calculate_features(self, new_trigger_node_id):
        target_class_node_index = [
            node for node in self.clean_set
            if self.clean_labels[node] == self.target_class
        ]
        etype = self.primary_type[0] + self.trigger_type[0]
        first_hops = set()
        for node in target_class_node_index:
            first_hop_etype = f"{self.primary_type[0]}{self.trigger_type[0]}"
            first_hop = self.hg.successors(node, etype=first_hop_etype).tolist()
            first_hops.update(first_hop)
        all_features = np.vstack([self.features[self.trigger_type][n] for n in first_hops])
        is_binary = np.isin(all_features, [0, 1]).all()
        if is_binary:
            new_feature = self.generate_binary_feature(all_features)
        else:
            new_feature = self.generate_continuous_feature(all_features)
        return torch.tensor(new_feature,dtype = self.hg.nodes[self.trigger_type].data['inp'].dtype).unsqueeze(dim=0)

    def generate_continuous_feature(self, all_features):

        if isinstance(all_features, torch.Tensor):
            all_features = all_features.cpu().numpy()
        _, num_features = all_features.shape

        sampled_values = []

        for i in range(num_features):
            feature_data = all_features[:, i]
            if np.all(feature_data == feature_data[0]):
                sampled_value = feature_data[0]
            else:
                try:
                    kde = gaussian_kde(feature_data)
                    sampled_value = kde.resample(size=1)[0]
                except np.linalg.LinAlgError:
                    mean_value = np.mean(feature_data)
                    noise = np.random.normal(0, 1e-6)
                    sampled_value = mean_value + noise

            sampled_values.append(sampled_value.item() if isinstance(sampled_value, np.ndarray) else sampled_value)

        sampled_array = np.array(sampled_values).flatten()
        return sampled_array
        # return np.mean(all_features, axis=0)


    def generate_binary_feature(self, all_features): 
        num_ones = np.sum(all_features, axis=0)
        num_zeros = all_features.shape[0] - num_ones
        prob_ones = num_ones / all_features.shape[0]
        new_features = np.random.rand(all_features.shape[1]) < prob_ones
        return new_features.astype(int)


    def search_nodes_by_degree(self, influential_numbers_by_type, two_hop_neighbors_by_type):
        selected_nodes_by_type = {}
        for node_type in influential_numbers_by_type.keys():
            n = influential_numbers_by_type[node_type]
            if node_type in two_hop_neighbors_by_type:
                if node_type == self.primary_type:
                    two_hop_neighbors_by_type[node_type] = {
                        node
                        for node in two_hop_neighbors_by_type[node_type]
                        if self.clean_labels[node] == self.target_class
                    }
                candidate_nodes = list(two_hop_neighbors_by_type[node_type])
            else:
                print(f"warning：Node type '{node_type}' doesn't exit")
                continue
            if len(candidate_nodes) < n:
                print(f"warning")
                n = len(candidate_nodes)
            node_degrees = []
            for node_id in candidate_nodes:
                total_deg = get_total_degree(self.hg, node_id, node_type)
                node_degrees.append((node_id, total_deg))
            sorted_nodes = sorted(node_degrees, key=lambda x: x[1], reverse=True)
            top_n_nodes = [node_id for node_id, deg in sorted_nodes[:n]]
            selected_nodes_by_type[node_type] = top_n_nodes
        return selected_nodes_by_type


    def select_influential_nodes(self):
        target_class_node_index = [
            node for node in range(len(self.clean_labels))
            if self.clean_labels[node] == self.target_class
        ]

        influential_two_hop_etypes = [
            etype for etype in self.hg.etypes
            if etype.startswith(self.trigger_type[0])
        ]

        two_hop_neighbors_by_type = {}
        influential_numbers_by_type = {}
        first_hops = set()
        second_dst_types = []
        for etype in influential_two_hop_etypes:
            src_type, dst_type = get_src_dst_from_etype(self.hg, etype=etype)
            second_dst_types.append(dst_type)
            if dst_type not in two_hop_neighbors_by_type:
                two_hop_neighbors_by_type[dst_type] = set()

            for node in target_class_node_index:
                first_hop_etype = f"{self.primary_type[0]}{self.trigger_type[0]}"
                first_hop = self.hg.successors(node, etype=first_hop_etype).tolist()

                for neighbor in first_hop:
                    first_hops.add(neighbor)
                    second_hop = self.hg.successors(neighbor, etype=etype).tolist()
                    for second_neighbor in second_hop:
                        two_hop_neighbors_by_type[dst_type].add(second_neighbor)

        for key in two_hop_neighbors_by_type:
            two_hop_neighbors_by_type[key] = list(two_hop_neighbors_by_type[key])

        trigger_nodes = self.hg.nodes(self.trigger_type)
        influential_numbers_by_type = {}
        for t in second_dst_types:
            total_edges = 0
            count = 0
            for trig_node in trigger_nodes:
                neighbors = self.hg.successors(trig_node, etype=self.trigger_type[0] + t[0])
                total_edges += len(neighbors)
                count += 1

            if count > 0:
                avg_degree = total_edges / count
            else:
                avg_degree = 1 

            influential_numbers_by_type[t] = math.ceil(avg_degree)
            if t == self.primary_type:
                influential_numbers_by_type[t] = influential_numbers_by_type[t]- 1

        return influential_numbers_by_type, two_hop_neighbors_by_type