import copy
import math
import numpy as np
from utils import get_src_dst_from_etype, get_total_degree
from ChosenPoisonNodesMethods.SecondConnect import create_victim_model
from scipy.stats import gaussian_kde
from tqdm import tqdm
import torch
from .backdoor import Backdoor


class CGBA(Backdoor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.the_method_search_influential_nodes = args[0].chosen_influential_nodes_method

    def construct_posion_graph(self, training = True):
        posion_hg = copy.deepcopy(self.hg)
        GREEN_TEXT = "\033[92m"
        RESET = "\033[0m"
        clean_labels_primary = self.clean_labels
        num_poison = len(self.posion_trainset_index)
        all_nodes = list(range(len(self.clean_labels)))
        if training:
            target_nodes = self.posion_trainset_index
        else:
            target_nodes = self.posion_testset_index


        primary_node_ids = posion_hg.nodes(self.primary_type)
        in_deg_sum = torch.zeros(len(primary_node_ids), dtype=torch.long)
        out_deg_sum = torch.zeros(len(primary_node_ids), dtype=torch.long)

        for etype in posion_hg.canonical_etypes:
            src_t, rel_t, dst_t = etype

            if dst_t == self.primary_type:
                in_degs_for_this_etype = posion_hg.in_degrees(primary_node_ids, etype=etype)
                in_deg_sum += in_degs_for_this_etype

            if src_t == self.primary_type:
                out_degs_for_this_etype = posion_hg.out_degrees(primary_node_ids, etype=etype)
                out_deg_sum += out_degs_for_this_etype

        total_degs_for_all_primary_nodes = in_deg_sum + out_deg_sum

        idx_map = {primary_node_ids[i].item(): i for i in range(len(primary_node_ids))}

        def get_degree_of(nid):
            idx_in_tensor = idx_map[nid] 
            return total_degs_for_all_primary_nodes[idx_in_tensor].item()

        max_node = max(all_nodes, key=lambda nid: get_degree_of(nid))

        trigger_size = self.args.CGBA_trigger_size

        posion_features = posion_hg.nodes[self.primary_type].data['inp']
        max_node_feat = posion_features[max_node]  

        sorted_indices = torch.argsort(max_node_feat, descending=True)
        self.topk_indices = sorted_indices[:trigger_size]
        self.trigger_value = max_node_feat[self.topk_indices]

        remain_target_nodes = [n for n in target_nodes if n != max_node]
        num_poison = min(num_poison, len(remain_target_nodes))

        poison_count = 0
        for node_id in remain_target_nodes:
            if poison_count >= num_poison:
                break
            posion_features[node_id, self.topk_indices] = self.trigger_value
            poison_count += 1
        posion_hg.nodes[self.primary_type].data['inp'] = posion_features

        return posion_hg












