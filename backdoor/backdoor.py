from abc import ABC, abstractmethod
import copy
from tqdm import tqdm
import torch
#------debug----

class Backdoor:
    def __init__(self, args, features, device, hg, hete_adjs, clean_labels, poison_labels, num_classes, posion_trainset_index, posion_testset_index, \
        clean_trainset_index, clean_testset_index, trainset_index, testset_index, primary_type, metapaths,text_attribute, edge_template, homo_g, node_mapping):
        self.args = args
        self.trigger_type = args.trigger_type 
        self.target_class = args.target_class 
        self.device = device
        self.hg = hg
        self.hete_adjs = hete_adjs
        self.features = features
        self.poison_labels = poison_labels 
        self.clean_labels = clean_labels 
        self.num_classes = num_classes
        self.posion_trainset_index = posion_trainset_index
        self.posion_testset_index = posion_testset_index
        self.clean_trainset_index = clean_trainset_index
        self.clean_testset_index = clean_testset_index
        self.clean_set = clean_trainset_index + self.clean_testset_index
        self.trainset_index = trainset_index
        self.testset_index = testset_index
        self.primary_type = primary_type 
        self.metapaths = metapaths
        self.text_attribute = text_attribute
        self.edge_template = edge_template
        self.homo_g = homo_g
        self.node_mapping = node_mapping
        self.reversed_node_mapping = {v: k for k, v in node_mapping.items()}
        self.the_method_search_influential_nodes = args.chosen_influential_nodes_method


    @abstractmethod
    def construct_posion_graph(self, training = True):
        pass

