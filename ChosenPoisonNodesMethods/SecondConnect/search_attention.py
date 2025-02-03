from docutils.nodes import target
from tqdm import tqdm
from .search_base import SearchSecondConnectNode
from victim_models.victim_models import SimpleHGNModel
from collections import defaultdict
from .utils import draw_long_tail_distribution_for_hetero, analysis_attention_degree
import random
import torch

class SearchNodesByAttention(SearchSecondConnectNode):
    def __init__(self, the_method_search_influential_nodes, influential_numbers_by_type,
                 two_hop_neighbors_by_type, primary_type, hg, target_class, clean_labels,
                 text_attribute, edge_template, metapaths, trainset_index, testset_index, args):
        super().__init__(the_method_search_influential_nodes, influential_numbers_by_type,
                         two_hop_neighbors_by_type, primary_type, hg, target_class, clean_labels)
        self.candidate_nodes_by_type = {}
        self.initial_pool_by_type = {} 
        self.text_attribute = text_attribute
        self.edge_template = edge_template
        self.metapaths = metapaths
        self.trainset_index = trainset_index
        self.testset_index = testset_index
        self.args = args
        self.hetero_attention = self.prepare_candidate_nodes()

    def get_node_attention(self, model, graph):
        device = self.args.device
        attentions = model.get_attention()
        if len(attentions) < 2:
            raise RuntimeError("Error: The number of GAT layer less than 2, Unable to get the two hop neighbors attention score.")

        att_l1 = attentions[0].to(device)  
        att_l2 = attentions[1].to(device)  

        edges_src, edges_dst = graph.edges()
        edges_src = edges_src.to(device)
        edges_dst = edges_dst.to(device)

        alpha_l1 = att_l1.mean(dim=1)
        alpha_l2 = att_l2.mean(dim=1)

        edge_att_l1 = {}
        edge_att_l2 = {}
        for eid in range(len(edges_src)): 
            u = edges_src[eid].item()
            v = edges_dst[eid].item()
            edge_att_l1[(u, v)] = alpha_l1[eid].item()
            edge_att_l2[(u, v)] = alpha_l2[eid].item()


        primary_shift = model.node_shift[self.primary_type]
        all_primary_num = self.hg.num_nodes(self.primary_type)
        target_nodes = []
        for local_id in range(all_primary_num):
            if self.clean_labels[local_id] == self.target_class:
                global_id = primary_shift + local_id
                target_nodes.append(global_id)

        node_twohop_dict={}
        for A in tqdm(target_nodes, desc = "Calculate Attention....."):
            first_hop_list = []
            srcs, dsts = graph.in_edges(A)
            for i in range(len(srcs)):
                u_ = srcs[i].item()
                v_ = dsts[i].item()  
                if (u_, A) in edge_att_l2:
                    first_hop_list.append(u_)

            twohop_importance = {}
            for v in first_hop_list: 

                att_vA = edge_att_l2.get((v, A), 0.0) 

                srcs2, dsts2 = graph.in_edges(v)
                for j in range(len(srcs2)):
                    w_ = srcs2[j].item()
                    att_wv = edge_att_l1.get((w_, v), 0.0)
                    if att_wv <= 0:
                        continue

                    imp_value = att_wv * att_vA

                    if w_ not in twohop_importance:
                        twohop_importance[w_] = 0.0
                    twohop_importance[w_] += imp_value

                node_twohop_dict[A] = {
                    k: v for k, v in sorted(twohop_importance.items(), key=lambda x: x[1], reverse=True)
                }

        return node_twohop_dict

    def aggregate_total_importance(self, node_twohop_dict):

        node_importance = defaultdict(float)
        for A_id, w_dict in node_twohop_dict.items():
            for w_id, imp_value in w_dict.items():
                node_importance[w_id] += imp_value
        return dict(node_importance)

    def map_global_id_to_hetero_node(self, model, hg, global_id):

        for ntype in model.node_shift:
            start_id = model.node_shift[ntype]
            count = hg.nodes[ntype].data['inp'].shape[0]  
            end_id = start_id + count
            if start_id <= global_id < end_id:
                local_id = global_id - start_id
                return (ntype, local_id)
        return (None, None)

    def map_importance_to_hetero(self, node_importance, model, hg):
        hetero_importance = defaultdict(list)
        for w_global_id, score in node_importance.items():
            ntype, local_id = self.map_global_id_to_hetero_node(model, hg, w_global_id)
            if ntype is not None:
                hetero_importance[ntype].append((local_id, score))

        sorted_hetero = {}
        for ntype, node_list in hetero_importance.items():
            sorted_nodes = sorted(node_list, key=lambda x: x[1], reverse=True)
            sorted_hetero[ntype] = sorted_nodes

        return sorted_hetero

    def analyze_twohop_importance(self, node_twohop_dict, model, hg):
        node_importance = self.aggregate_total_importance(node_twohop_dict)
        sorted_hetero = self.map_importance_to_hetero(node_importance, model, hg)
        return sorted_hetero


    def prepare_candidate_nodes(self):
        num_classes = max(self.clean_labels).item() + 1
        model = SimpleHGNModel(self.args, self.hg, self.text_attribute, self.clean_labels, num_classes,
                       self.primary_type, self.metapaths, self.trainset_index, self.testset_index,
                       self.testset_index)
        model.train_victim_model()
        node_twohop_dict  = self.get_node_attention(model, model.g)
        self.node_two_hope_dict = node_twohop_dict
        sorted_hetero = self.analyze_twohop_importance(node_twohop_dict, model, self.hg)

        draw_long_tail_distribution_for_hetero(
            hetero_attention=sorted_hetero,
            tail_quantile=0.98,  
            use_kde=True,
            log_scale=False
        )
        self.analysis_result = analysis_attention_degree(sorted_hetero,self.hg, self.target_class, self.clean_labels, self.primary_type, self.args.trigger_type, self.args.dataset)
        return sorted_hetero

    def get_top_n_nodes(self, cooresponding_attention,n, target_node, node_type):
        top_n_nodes = []
        for item in cooresponding_attention:
            if target_node == item[0] and node_type == self.primary_type:
                continue
            else:
                top_n_nodes.append(item[0])
            if len(top_n_nodes)>=n:
                break
        return top_n_nodes

    def getFilterNodes(self, target_node=None):
        if target_node is None:
            raise ValueError("target node is required in this subclass.")
        selected_nodes_by_type = {}
        for node_type in self.influential_numbers_by_type.keys():
            n = self.influential_numbers_by_type[node_type]
            cooresponding_attention = self.hetero_attention[node_type]
            selected_nodes = self.get_top_n_nodes(cooresponding_attention, n, target_node, node_type)
            selected_nodes_by_type[node_type] = selected_nodes

        return selected_nodes_by_type


