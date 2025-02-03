import numpy as np
import networkx as nx
import copy
import random
import pickle
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import dgl
from ChosenPoisonNodesMethods.PoisonTrainsetSelect import *

def select_random_items(dictionary, n):
    return dict(random.sample(list(dictionary.items()), n))

def find_dict_difference(dict1, dict2):
    return {key: value for key, value in dict1.items() if key not in dict2}



def split_index(hg, primary_type, labels, text_attribute, args, **kwargs):
    chosen_poison_nodes_method = args.chosen_poison_nodes_method
    target_class = args.target_class
    poison_set_ratio = args.poison_set_ratio
    backdoor_type = args.backdoor_type
    test_set_ratio = 0.2
    validation_set_ratio = 0.1
    homo_g, node_mapping = heterograph_to_networkx(hg, primary_type)

    if backdoor_type == 'CGBA':
        target_class_nodes = [i for i, label in enumerate(labels) if label == target_class]
        non_target_class_nodes = [i for i, label in enumerate(labels) if label != target_class]
        num_total_nodes = int(labels.shape[0])

        num_train_poison_nodes = int(poison_set_ratio * num_total_nodes)//2
        if num_train_poison_nodes > len(target_class_nodes):
            num_train_poison_nodes = len(target_class_nodes)
        num_test_poison_nodes = int(poison_set_ratio * num_total_nodes)//2

        selector = RandomSelector(homo_g, node_mapping, primary_type)
        poison_trainset_index = selector.select_poison_nodes(target_class_nodes, num_train_poison_nodes)

        poison_testset_index = list(np.random.choice(non_target_class_nodes, num_test_poison_nodes, replace=False))
        poison_nodes = poison_testset_index + poison_trainset_index

        all_nodes = list(range(len(labels)))
        remaining_nodes =  [n for n in all_nodes if n not in poison_nodes]
        num_test_nodes = int(test_set_ratio * num_total_nodes)
        num_clean_test_nodes = num_test_nodes - len(poison_trainset_index)
        clean_testset_index = list(np.random.choice(remaining_nodes, num_clean_test_nodes, replace=False))
        testset_index = clean_testset_index + poison_testset_index

        remianing_nodes = [n for n in all_nodes if n not in poison_nodes + testset_index]
        num_validation_nodes = int(validation_set_ratio * num_total_nodes)
        validation_set_index = list(np.random.choice(remianing_nodes, num_validation_nodes, replace=False))

        clean_trainset_index = [n for n in all_nodes if n not in poison_nodes + testset_index + validation_set_index]
        trainset_index = clean_trainset_index + poison_trainset_index
        return poison_trainset_index, poison_testset_index, clean_trainset_index, clean_testset_index, trainset_index, testset_index, validation_set_index, labels, homo_g, node_mapping

    else:
        non_target_class_nodes = [i for i, label in enumerate(labels) if label != target_class]

        num_total_nodes = int(labels.shape[0])
        num_train_poison_nodes = int(poison_set_ratio * num_total_nodes)//2 
        num_test_poison_nodes = int(poison_set_ratio * num_total_nodes) - num_train_poison_nodes

        if chosen_poison_nodes_method == 'random':
            selector = RandomSelector(homo_g, node_mapping, primary_type)
            poison_trainset_index = selector.select_poison_nodes(non_target_class_nodes, num_train_poison_nodes)

        elif chosen_poison_nodes_method == 'degree':
            selector = DegreeBasedSelector(homo_g, node_mapping, primary_type)
            poison_trainset_index = selector.select_poison_nodes(non_target_class_nodes, num_train_poison_nodes)

        elif chosen_poison_nodes_method == 'pagerank':
            selector = PageRankBasedSelector(homo_g, node_mapping, primary_type)
            poison_trainset_index = selector.select_poison_nodes(non_target_class_nodes, num_train_poison_nodes)

        elif chosen_poison_nodes_method == 'llm':
            selector = LLMBasedSelector(homo_g, node_mapping, primary_type, text_attribute)
            poison_trainset_index = selector.select_poison_nodes(non_target_class_nodes, num_train_poison_nodes)

        elif chosen_poison_nodes_method == 'cluster':
            selector = ClusterBasedSelector(homo_g, node_mapping, labels, primary_type, text_attribute, hg, args, **kwargs)
            poison_trainset_index = selector.select_poison_nodes(non_target_class_nodes, num_train_poison_nodes)

        else:
            raise ValueError(f"Unsupported poison nodes selection method: {chosen_poison_nodes_method}")

        targte_remaining_nodes = [i for i in non_target_class_nodes if i not in poison_trainset_index]
        poison_testset_index =list(np.random.choice(targte_remaining_nodes, num_test_poison_nodes, replace=False))
        poison_nodes = poison_testset_index + poison_trainset_index

        remaining_nodes = [i for i in range(labels.shape[0]) if i not in poison_nodes]
        num_test_nodes = int(test_set_ratio * num_total_nodes) - len(poison_testset_index)
        clean_testset_index = list(np.random.choice(remaining_nodes, num_test_nodes, replace=False))

        remaining_train_nodes = [i for i in range(labels.shape[0]) if i not in poison_nodes and i not in clean_testset_index]
        num_validation_nodes = int(validation_set_ratio * num_total_nodes)
        validation_set_index = list(np.random.choice(remaining_train_nodes, num_validation_nodes, replace=False))

        clean_trainset_index = [i for i in remaining_train_nodes if i not in validation_set_index]

        trainset_index = clean_trainset_index + poison_trainset_index
        testset_index = clean_testset_index + poison_testset_index

        poison_set_index = poison_trainset_index + poison_testset_index
        labels[poison_set_index] = target_class

        return poison_trainset_index, poison_testset_index, clean_trainset_index, clean_testset_index, trainset_index, testset_index, validation_set_index, labels, homo_g, node_mapping

def heterograph_to_networkx(g,primary_type):
    nx_graph = nx.DiGraph()
    node_mapping = {}
    types = copy.deepcopy(g.ntypes)
    types.remove(primary_type)
    types.insert(0, primary_type)

    node_id = 0
    for ntype in types:
        num_nodes = g.number_of_nodes(ntype)
        for nid in range(num_nodes):
            node_mapping[(ntype, nid)] = node_id
            nx_graph.add_node(node_id)
            node_id += 1

    for etype in g.canonical_etypes:
        src, dst = g.edges(etype=etype)
        src = src.numpy()
        dst = dst.numpy()

        for s, d in zip(src, dst):
            nx_src = node_mapping[(etype[0], s)]
            nx_dst = node_mapping[(etype[2], d)]
            nx_graph.add_edge(nx_src, nx_dst)

    return nx_graph, node_mapping


def get_src_dst_from_etype(graph, etype):
    for canonical_etype in graph.canonical_etypes:
        src_type, edge_type, dst_type = canonical_etype
        if edge_type == etype:
            return src_type, dst_type
    return None, None


def get_total_degree(hg, node_id, node_type):
    in_deg = 0
    in_edge_types = [etype for etype in hg.canonical_etypes if etype[2] == node_type]
    for etype in in_edge_types:
        deg = hg.in_degrees(node_id, etype=etype)
        in_deg += deg

    out_deg = 0
    out_edge_types = [etype for etype in hg.canonical_etypes if etype[0] == node_type]
    for etype in out_edge_types:
        deg = hg.out_degrees(node_id, etype=etype)
        out_deg += deg

    total_deg = in_deg + out_deg
    return total_deg  

def get_pagerank(hg, node_id, node_type):
    alpha = 0.85
    max_iter = 100
    tol = 1e-6

    num_nodes = hg.num_nodes(node_type)
    scores = torch.ones(num_nodes) / num_nodes

    for _ in range(max_iter):
        old_scores = scores.clone()
        scores = torch.ones(num_nodes) / num_nodes * (1 - alpha)

        for etype in hg.etypes:
            src_type, _, dst_type = hg.to_canonical_etype(etype)
            if dst_type != node_type:
                continue

            degs = hg.out_degrees(etype=etype).float()
            degs = torch.where(degs > 0, degs, torch.ones_like(degs))

            if src_type == node_type:
                src_scores = old_scores
            else:
                src_scores = torch.ones(hg.num_nodes(src_type)) / hg.num_nodes(src_type)

            normalized_scores = src_scores / degs

            with hg.local_scope():
                hg.nodes[src_type].data['h'] = normalized_scores
                hg.update_all(
                    message_func=dgl.function.copy_u('h', 'm'),
                    reduce_func=dgl.function.sum('m', 'h'),
                    etype=etype
                )
                scores += alpha * hg.nodes[node_type].data['h']

        if torch.abs(scores - old_scores).sum() < tol:
            break

    return scores[node_id].item()



def remove_intersection(list1, list2):
    return [item for item in list1 if item not in list2]

def update_hg_features(hg,args,features):
    for ntype in hg.ntypes:
        hg.nodes[ntype].data['inp'] = features[ntype].clone().detach().float()
    return hg

class EarlyStopping:
    def __init__(self, patience, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score - self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
