import copy
import math
import numpy as np
from utils import get_src_dst_from_etype, get_total_degree
from ChosenPoisonNodesMethods.SecondConnect import create_victim_model
from scipy.stats import gaussian_kde
from tqdm import tqdm
import networkx as nx
from .backdoor import Backdoor
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from copy import deepcopy
from .GCN import GCN

class GradWhere(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, thrd, device):
        ctx.save_for_backward(input)
        rst = torch.where(input > thrd,
                          torch.tensor(1.0, device=device, requires_grad=True),
                          torch.tensor(0.0, device=device, requires_grad=True))
        return rst

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        return grad_input, None, None


class GraphTrojanNet(nn.Module):
    def __init__(self, device, nfeat, nout, layernum=1, dropout=0.00):
        super(GraphTrojanNet, self).__init__()

        layers = []
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
        for l in range(layernum - 1):
            layers.append(nn.Linear(nfeat, nfeat))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))

        self.layers = nn.Sequential(*layers).to(device)
        self.feat = nn.Linear(nfeat, nout * nfeat)
        self.edge = nn.Linear(nfeat, int(nout * (nout - 1) / 2))
        self.device = device

    def forward(self, input, thrd):
        GW = GradWhere.apply
        h = self.layers(input)
        feat = self.feat(h)
        edge_weight = self.edge(h)
        edge_weight = GW(edge_weight, thrd, self.device)
        return feat, edge_weight


class HomoLoss(nn.Module):
    def __init__(self, args, device):
        super(HomoLoss, self).__init__()
        self.args = args
        self.device = device

    def forward(self, trigger_edge_index, trigger_edge_weights, x, thrd):
        trigger_edge_index = trigger_edge_index[:, trigger_edge_weights > 0.0]
        edge_sims = F.cosine_similarity(x[trigger_edge_index[0]], x[trigger_edge_index[1]])
        loss = torch.relu(thrd - edge_sims).mean()
        return loss


class UGBA(Backdoor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trigger_index = self.get_trigger_index(self.args.UGBA_trigger_size)
        self.train_trigger_generator()

    def get_trigger_index(self, trigger_size): 
        edge_list = []
        edge_list.append([0, 0])
        for j in range(trigger_size):
            for k in range(j):
                edge_list.append([j, k])
        edge_index = torch.tensor(edge_list, device=self.device).long().T
        return edge_index

    def get_trojan_edge(self, start, idx_attach, trigger_size):
        edge_list = []
        for idx in idx_attach:
            edges = self.trigger_index.clone()
            edges[0, 0] = idx
            edges[1, 0] = start
            edges[:, 1:] = edges[:, 1:] + start
            edge_list.append(edges)
            start += trigger_size

        edge_index = torch.cat(edge_list, dim=1)
        row = torch.cat([edge_index[0], edge_index[1]])
        col = torch.cat([edge_index[1], edge_index[0]])
        edge_index = torch.stack([row, col])
        return edge_index

    def fit(self, model, optimizer, training=True):
        # Initialize components
        if not hasattr(self, 'trojan'):
            self.trojan = GraphTrojanNet(
                self.device,
                self.features[self.primary_type].shape[1],
                self.args.UGBA_trigger_size,
                layernum=2
            ).to(self.device) 
            self.homo_loss = HomoLoss(self.args, self.device)

        # Get indices
        idx_attach = self.posion_trainset_index if training else self.posion_testset_index 
        idx_train = self.trainset_index

        homo_attach = [self.node_mapping[(self.primary_type, idx)] for idx in idx_attach]
        homo_train = [self.node_mapping[(self.primary_type, idx)] for idx in idx_train]

        idx_attach = torch.tensor(homo_attach, device=self.device)
        idx_train = torch.tensor(homo_train, device=self.device)

        # Get graph components
        edge_index = torch.tensor(list(self.homo_g.edges())).T.to(self.device) 
        edge_weight = torch.ones(edge_index.shape[1], device=self.device) 

        # Create features tensor
        reverse_mapping = {het_idx: het_type for het_type, het_idx in self.node_mapping.items()}

        features = []
        for node in range(self.homo_g.number_of_nodes()):
            het_type = reverse_mapping[node]
            node_type, node_id = het_type
            features.append(self.features[node_type][node_id])

        features = torch.stack(features).to(self.device)

        # Initialize shadow model
        self.shadow_model = type(model)(
            nfeat=features.shape[1],
            nhid=self.args.UGBA_hidden,
            nclass=self.num_classes,
            dropout=0.0,
            device=self.device
        ).to(self.device)

        optimizer_shadow = optim.Adam(self.shadow_model.parameters(), lr=self.args.UGBA_lr)
        optimizer_trigger = optim.Adam(self.trojan.parameters(), lr=self.args.UGBA_lr)

        # Training process
        loss_best = float('inf')
        for epoch in range(self.args.UGBA_trojan_epochs):
            self.trojan.train()

            # Inner optimization
            for _ in range(self.args.UGBA_inner):
                optimizer_shadow.zero_grad()

                trojan_feat, trojan_weights = self.trojan(features[idx_attach].float(), self.args.UGBA_thrd)
                trojan_weights = torch.cat([
                    torch.ones([len(trojan_feat), 1], dtype=torch.float, device=self.device),
                    trojan_weights
                ], dim=1).flatten() 

                trojan_feat = trojan_feat.view([-1, features.shape[1]]) 
                trojan_edge = self.get_trojan_edge(len(features), idx_attach, self.args.UGBA_trigger_size)
                poison_edge_weights = torch.cat([edge_weight, trojan_weights, trojan_weights]).detach()
                poison_x = torch.cat([features, trojan_feat]).detach().float()
                poison_edge_index = torch.cat([edge_index, trojan_edge], dim=1)

                output = self.shadow_model(poison_x, poison_edge_index, poison_edge_weights)

                poison_labels = self.clean_labels.clone().to(self.device)
                poison_labels[idx_attach] = self.target_class

                loss_inner = F.nll_loss(
                    output[idx_train],
                    poison_labels[idx_train]
                )

                loss_inner.backward()
                optimizer_shadow.step()

            # Outer optimization
            self.trojan.eval()
            optimizer_trigger.zero_grad()

            # Sample unlabeled nodes
            clean_train_homo_indices = [
                self.node_mapping[(self.primary_type, idx)]
                for idx in self.clean_trainset_index
            ]
            rs = np.random.RandomState(self.args.random_seed)
            sampled_unlabeled = rs.choice(
                clean_train_homo_indices,
                size=min(512, len(clean_train_homo_indices)),
                replace=False
            )
            idx_outer = torch.cat([
                idx_attach,
                torch.tensor(sampled_unlabeled, device=self.device)
            ])

            trojan_feat, trojan_weights = self.trojan(features[idx_outer].float(), self.args.UGBA_thrd) 
            trojan_weights = torch.cat([
                torch.ones([len(idx_outer), 1], dtype=torch.float, device=self.device),
                trojan_weights
            ], dim=1).flatten()


            trojan_feat = trojan_feat.view([-1, features.shape[1]]) 
            trojan_edge = self.get_trojan_edge(len(features), idx_outer, self.args.UGBA_trigger_size)

            update_edge_weights = torch.cat([edge_weight, trojan_weights, trojan_weights])
            update_feat = torch.cat([features, trojan_feat]).float()
            update_edge_index = torch.cat([edge_index, trojan_edge], dim=1)

            output = self.shadow_model(update_feat, update_edge_index, update_edge_weights)

            labels_outer = self.clean_labels.clone().to(self.device)
            labels_outer[idx_outer] = self.target_class

            loss_target = self.args.UGBA_target_loss_weight * F.nll_loss(
                output[torch.cat([idx_train])],
                labels_outer[idx_train]
            )

            loss_homo = 0.0
            if self.args.UGBA_homo_loss_weight > 0:
                loss_homo = self.homo_loss(
                    trojan_edge[:, :int(trojan_edge.shape[1] / 2)],
                    trojan_weights,
                    update_feat,
                    self.args.UGBA_homo_boost_thrd
                )

            loss_outer = loss_target + self.args.UGBA_homo_loss_weight * loss_homo
            loss_outer.backward()
            optimizer_trigger.step()

            if loss_outer < loss_best:
                self.weights = deepcopy(self.trojan.state_dict())
                loss_best = float(loss_outer)

            if epoch % 10 == 0:
                print(f'Epoch {epoch}:')
                print(f'Inner Loss: {loss_inner:.5f}, Target Loss: {loss_target:.5f}, Homo Loss: {loss_homo:.5f}')

        self.trojan.load_state_dict(self.weights)
        self.trojan.eval()

    def train_trigger_generator(self, training=True):
        shadow_model = GCN(
            nfeat=self.features[self.primary_type].shape[1],
            nhid=self.args.UGBA_hidden,
            nclass=self.num_classes,
            dropout=0.5,
            device=self.device
        ).to(self.device)

        # Create optimizer for shadow model
        shadow_optimizer = optim.Adam(shadow_model.parameters(), lr=self.args.UGBA_lr)

        # Train the trigger generator
        self.fit(shadow_model, shadow_optimizer, training)

    def construct_posion_graph(self, training=True):
        print("Constructing poisoned graph using UGBA")
        with torch.no_grad():
            # Get nodes to poison
            poison_nodes = self.posion_trainset_index if training else self.posion_testset_index

            # Convert to homogeneous indices
            homo_poison_nodes = []
            for node_id in poison_nodes:
                homo_key = (self.primary_type, node_id)
                if homo_key in self.node_mapping:
                    homo_poison_nodes.append(self.node_mapping[homo_key])

            # Create features tensor
            features = []
            reverse_mapping = {het_idx: het_type for het_type, het_idx in self.node_mapping.items()}
            for node in range(self.homo_g.number_of_nodes()):
                het_type = reverse_mapping[node]
                node_type, node_id = het_type
                features.append(self.features[node_type][node_id])
            features = torch.stack(features).to(self.device)

            # Generate triggers using trained generator
            idx_attach = torch.tensor(homo_poison_nodes, device=self.device)
            trojan_feat, trojan_weights = self.trojan(features[idx_attach].float(), self.args.UGBA_thrd)

            # Process trigger weights
            trojan_weights = torch.cat([
                torch.ones([len(trojan_feat), 1], dtype=torch.float, device=self.device),
                trojan_weights
            ], dim=1)
            n = trojan_weights.shape[0]
            m = self.args.UGBA_trigger_size

            trojan_type = []
            node_types = deepcopy(self.hg.ntypes)
            node_types.remove(self.primary_type)
            for _ in range(n):
                row = np.random.choice(node_types, size= 1).tolist()

                random_types = np.random.choice(node_types, size=m - 1).tolist()
                row.extend(random_types)

                trojan_type.append(row)

            trojan_feat = trojan_feat.view(len(idx_attach), self.args.UGBA_trigger_size, -1)
            poison_hg = deepcopy(self.hg)
            legal_etypes = self.hg.canonical_etypes
            legal_edges = []
            trigger_nodes_mapping = {}  
            current_node_count = {ntype: len(self.hg.nodes(ntype)) for ntype in self.hg.ntypes}
            for idx, (types, weights) in enumerate(zip(trojan_type, trojan_weights)):
                poison_node = self.reversed_node_mapping[poison_nodes[idx]][1]  
                trigger_nodes = []  

                for i, ntype in enumerate(types):
                    new_id = current_node_count[ntype]
                    current_node_count[ntype] += 1
                    trigger_nodes.append((ntype, new_id))
                    node_feature = trojan_feat[idx, i] 
                    poison_hg.add_nodes(1,
                                        ntype=ntype,
                                        data={'inp': node_feature.unsqueeze(0).to(poison_hg.device)})

              
                src_type = self.primary_type
                dst_type = types[0]
                for s, e, d in legal_etypes:
                    if s == src_type and d == dst_type:
                        legal_edges.append((poison_node, trigger_nodes[0][1], e))
                    if s == dst_type and d == src_type:
                        legal_edges.append((trigger_nodes[0][1], poison_node, e))

                edge_idx = 1 
                for i in range(len(trigger_nodes)):
                    for j in range(i + 1, len(trigger_nodes)):
                        if weights[edge_idx] > 0:
                            src_type = types[i]
                            dst_type = types[j]
                            edge_found = False
                            for s, e, d in legal_etypes:
                                if (s == src_type and d == dst_type):
                                    legal_edges.append((trigger_nodes[i][1], trigger_nodes[j][1], e))
                                    edge_found = True
                                elif (s == dst_type and d == src_type):
                                    legal_edges.append((trigger_nodes[j][1], trigger_nodes[i][1], e))
                                    edge_found = True

                        edge_idx += 1

            for src, dst, etype in legal_edges:
                poison_hg.add_edges(src, dst, etype=etype)

            for ntype in poison_hg.ntypes:
                isolated_nodes = []
                num_nodes = poison_hg.number_of_nodes(ntype)  

                for nid in range(num_nodes):  
                    is_isolated = True
                    for src_type, etype, dst_type in poison_hg.canonical_etypes:
                        if dst_type == ntype:
                            in_degree = poison_hg.in_degrees(nid, etype=etype)
                            if in_degree > 0:
                                is_isolated = False
                                break

                        if src_type == ntype:
                            out_degree = poison_hg.out_degrees(nid, etype=etype)
                            if out_degree > 0:
                                is_isolated = False
                                break

                    if is_isolated:
                        isolated_nodes.append(nid)

                if isolated_nodes:
                    poison_hg.remove_nodes(isolated_nodes, ntype=ntype)

        return poison_hg




