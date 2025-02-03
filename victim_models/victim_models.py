import dgl
import copy
import torch
import torch.nn as nn
import numpy as np
import scipy.sparse as sp
import torch.nn.functional as F
from .utils import EarlyStopping
from .HGT.HGTModel import HGT
from .HAN.model import HAN
from .RGCN.RGCN import RGCN
from .SimpleHGN.GNN import myGAT
from sklearn.model_selection import train_test_split
import random


class VictimModel:
    def __init__(self, args, hg, text_attribute, labels, num_classes, primary_type, metapaths,
                 train_idx, val_idx, test_idx):
        self.args = args
        self.device = torch.device(args.device)
        self.dataset = args.dataset
        self.model = None
        self.hg = hg
        self.text_attribute = text_attribute
        self.labels = torch.LongTensor(labels)
        self.num_classes = num_classes
        self.primary_type = primary_type
        self.metapaths = metapaths
        self.train_idx = train_idx
        self.val_idx = val_idx
        self.test_idx = test_idx

    def create_model(self):
        raise NotImplementedError

    def predict(self, G):
        raise NotImplementedError

    def evaluate(self, G):
        raise NotImplementedError

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            gain = nn.init.calculate_gain('relu')
            nn.init.xavier_normal_(module.weight, gain=gain)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Parameter):
            nn.init.xavier_normal_(module)

    def train_victim_model(self):
        self.create_model()
        self.model.apply(self.init_weights)
        optimizer = self.get_optimizer() 
        scheduler = self.get_scheduler(optimizer) 
        
        best_val_metric = self.get_initial_best_metric() 
        best_model = None 
        early_stopping = self.get_early_stopping() 

        for epoch in np.arange(self.args.n_epoch) + 1:
            self.model.train() 
            optimizer.zero_grad()
            logits = self.forward(self.hg)
            loss = self.compute_loss(logits)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            if self.should_evaluate(epoch):
                val_metric = self.evaluate_model()
                if self.is_better_metric(val_metric, best_val_metric):
                    best_val_metric = val_metric
                    print("Updating best model....")
                    best_model = copy.deepcopy(self.model)
                self.log_epoch(epoch, optimizer, loss, val_metric, best_val_metric)

                if early_stopping is not None:
                    early_stopping(val_metric, self.model)
                    if early_stopping.early_stop:
                        print('Early stopping!')
                        break

        if best_model is not None:
            print("Save the best model....")
            self.model = best_model

    def get_optimizer(self):
        return torch.optim.AdamW(self.model.parameters())

    def get_scheduler(self, optimizer):
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer, total_steps=self.args.n_epoch, max_lr=self.args.max_lr
        )

    def get_initial_best_metric(self):
        # Default is accuracy, so initial best is 0
        return torch.tensor(0)

    def get_early_stopping(self):
        # Default is no early stopping
        return None

    def should_evaluate(self, epoch):
        # Evaluate every 5 epochs by default
        return epoch % 5 == 0

    def is_better_metric(self, current_metric, best_metric):
        # For accuracy, higher is better
        return current_metric > best_metric

    def forward(self, G):
        return self.predict(G)

    def compute_loss(self, logits):
        return F.cross_entropy(
            logits[self.train_idx], self.labels[self.train_idx].to(self.device)
        )

    def evaluate_model(self):
        self.model.eval()
        logits = self.predict(self.hg)
        pred = logits.argmax(1).cpu()
        val_acc = (pred[self.val_idx] == self.labels[self.val_idx]).float().mean()
        return val_acc

    def log_epoch(self, epoch, optimizer, loss, val_metric, best_val_metric):
        lr = optimizer.param_groups[0]["lr"] if optimizer is not None else 0
        print(
            "Epoch: %d LR: %.5f Loss %.4f, Validation Metric %.4f (Best %.4f)"
            % (
                epoch,
                lr,
                loss.item(),
                val_metric,
                best_val_metric,
            )
        )


class HANModel(VictimModel):
    def __init__(self, args, hg, text_attribute, labels, num_classes, primary_type, metapaths,
                 train_idx, val_idx, test_idx):
        super(HANModel, self).__init__(args, hg, text_attribute, labels, num_classes, primary_type, metapaths,
                                       train_idx, val_idx, test_idx)

    def create_model(self):
        self.hg = self.hg.to(self.device)
        self.model = HAN(
            meta_paths=self.metapaths,
            in_size=self.hg.nodes[self.primary_type].data["inp"].shape[1],
            hidden_size=self.args.HAN_n_hid,
            out_size=self.labels.max().item() + 1,
            num_heads=self.args.HAN_num_heads,
            dropout=self.args.HAN_dropout
        ).to(self.device)

    def predict(self, G, eval=False):
        if not eval:
            return self.model(G, G.ndata['inp'][self.primary_type])
        else:
            self.model.eval()
            return self.model(G, G.ndata['inp'][self.primary_type])

    def evaluate(self, G):
        G = G.to(self.device)
        return self.predict(G, eval=True)

class HGTModel(VictimModel):
    def __init__(self, args, hg, text_attribute, labels, num_classes, primary_type, metapaths,
                 train_idx, val_idx, test_idx):
        super(HGTModel, self).__init__(args, hg, text_attribute, labels, num_classes, primary_type,
                                       metapaths, train_idx, val_idx, test_idx)

    def create_model(self):
        node_dict = {}
        edge_dict = {}
        for ntype in self.hg.ntypes:
            node_dict[ntype] = len(node_dict)
        for etype in self.hg.etypes:
            edge_dict[etype] = len(edge_dict)
            self.hg.edges[etype].data["id"] = (
                torch.ones(self.hg.num_edges(etype), dtype=torch.long) * edge_dict[etype]
            )
        self.hg = self.hg.to(self.device)

        self.model = HGT(
            self.hg,
            node_dict,
            edge_dict,
            n_inp=self.hg.nodes[self.primary_type].data["inp"].shape[1],
            n_hid=self.args.HGT_n_hid,
            n_out=self.labels.max().item() + 1,
            n_layers=8,
            n_heads=4,
            use_norm=True,
        ).to(self.device)

    def predict(self, G, eval=False):
        if not eval:
            return self.model(G, self.primary_type)
        else:
            self.model.eval()
            return self.model(G, self.primary_type)

    def evaluate(self, G):
        node_dict = {}
        edge_dict = {}
        for ntype in G.ntypes:
            node_dict[ntype] = len(node_dict)
        for etype in G.etypes:
            edge_dict[etype] = len(edge_dict)
            G.edges[etype].data["id"] = (
                torch.ones(G.num_edges(etype), dtype=torch.long) * edge_dict[etype]
            )
        G = G.to(self.device)
        return self.predict(G, eval=True)


class RGCNModel(VictimModel):
    def __init__(self, args, hg, text_attribute, labels, num_classes, primary_type, metapaths,
                 train_idx, val_idx, test_idx):
        super(RGCNModel, self).__init__(args, hg, text_attribute, labels, num_classes, primary_type,
                                       metapaths, train_idx, val_idx, test_idx)

    def create_model(self):
        self.hg = self.hg.to(self.device)
        self.model = RGCN(in_feats=self.hg.nodes[self.primary_type].data["inp"].shape[1],
                     hidden_feats=self.args.RGCN_n_hid,
                     out_feats=self.labels.max().item() + 1,
                     rels=self.hg.etypes,
                     target_type=self.primary_type,
                     ).to(self.device)

    def predict(self, G, eval=False):
        if not eval:
            return self.model(G,G.ndata['inp'])
        else:
            self.model.eval()
            return self.model(G, G.ndata['inp'])

    def evaluate(self, G):
        G = G.to(self.device)
        return self.predict(G, eval=True)


class SimpleHGNModel(VictimModel):
    def __init__(
        self,
        args,
        hg,
        text_attribute,
        labels,
        num_classes,
        primary_type,
        metapaths,
        train_idx,
        val_idx,
        test_idx,
    ):
        super(SimpleHGNModel, self).__init__(
            args,
            hg,  # No need to pass features separately
            text_attribute,
            labels,
            num_classes,
            primary_type,
            metapaths,
            train_idx,
            val_idx,
            test_idx,
        )
        self.process_data(
            self.hg,
            self.labels,
            self.train_idx,
            self.val_idx,
            self.test_idx,
        )

    def get_optimizer(self):
        args = self.args
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=args.SimpleHGN_lr,
            weight_decay=args.SimpleHGN_weight_decay,
        )

    def get_scheduler(self, optimizer):
        # SimpleHGNModel does not use a scheduler
        return None

    def get_early_stopping(self):
        args = self.args
        return EarlyStopping(
            patience=args.SimpleHGN_patience,
            verbose=True,
        )


    def compute_loss(self, logits):
        logp = F.log_softmax(logits, dim=-1)
        train_loss = F.nll_loss(
            logp[self.train_idx_global],
            self.labels_tensor[self.train_idx_global - self.primary_shift],
        )
        return train_loss

    def evaluate_model(self):
        self.model.eval()
        with torch.no_grad():
            logits = self.model(self.features_list, self.e_feat)
            pred = logits.argmax(1).cpu()
            val_acc = (
                pred[self.val_idx_global]
                == self.labels_tensor[self.val_idx_global - self.primary_shift].cpu()
            ).float().mean()
        return val_acc.item()

    def forward(self, G):
        return self.model(self.features_list, self.e_feat)

    def get_embeddings(self):
        self.model.eval()
        logits = self.model(self.features_list, self.e_feat)
        embeddings = self.model.embeddings
        return embeddings[self.primary_global_ids]

    def get_specific_embeddings(self, two_hop_neighbors_by_type):
        self.model.eval()
        logits = self.model(self.features_list, self.e_feat)
        embeddings = self.model.embeddings
        corresponding_embeddings = {}
        for type in two_hop_neighbors_by_type.keys():
            nodes = list(two_hop_neighbors_by_type[type])
            node_index = list(map(lambda x: x + self.node_shift[type], nodes))
            corresponding_embeddings[type] = embeddings[node_index]

        return corresponding_embeddings


    def log_epoch(self, epoch, optimizer, loss, val_metric, best_val_metric):
        print(
            "Epoch {:05d} | Train_Loss: {:.4f} | Val_Acc {:.4f} (Best {:.4f})".format(
                epoch, loss.item(), val_metric, best_val_metric
            )
        )

    def process_data(self, hg, labels, train_idx, val_idx, test_idx):
        node_types = list(hg.ntypes)
        features = {ntype: hg.nodes[ntype].data['inp'] for ntype in node_types}
        (
            self.g,
            self.features_list,
            self.labels_tensor,
            self.e_feat,
            self.num_etypes,
            self.in_dims,
            self.num_classes,
            self.primary_shift,
            self.primary_global_ids,
        ) = self.process_graph(hg, features, labels)

        # Adjust train_idx, val_idx, test_idx
        self.train_idx_global = self.primary_global_ids[train_idx]
        self.val_idx_global = self.primary_global_ids[val_idx]
        self.test_idx_global = self.primary_global_ids[test_idx]

    def process_graph(self, G, features, labels):
        node_types = list(G.ntypes) 
        node_type2id = {node_type: idx for idx, node_type in enumerate(node_types)}

        # Step 2: Calculate node counts and shifts
        node_counts = {
            node_type: features[node_type].shape[0] for node_type in node_types
        }
        nodes_shift = {}
        shift = 0
        for node_type in node_types:
            nodes_shift[node_type] = shift
            shift += node_counts[node_type]
        total_num_nodes = shift  # Total number of nodes
        self.node_shift = nodes_shift

        # Step 3: Create features_list
        features_list = []
        for node_type in node_types:
            feat = features[node_type].float()
            features_list.append(feat.to(self.device))

        # Step 4: Adjust adjacency matrices and create adjM
        edge_types = [etype[1] for etype in G.canonical_etypes] 
        edge_type_to_nodes = {
            etype[1]: (etype[0], etype[2]) for etype in G.canonical_etypes
        }

        adjM_list = []
        for etype in edge_types:
            src_nodes, dst_nodes = G.edges(etype=etype)
            src_nodes = src_nodes.numpy()
            dst_nodes = dst_nodes.numpy()
            src_node_type, dst_node_type = edge_type_to_nodes[etype] 
            src_shift = nodes_shift[src_node_type]
            dst_shift = nodes_shift[dst_node_type]

            src = src_nodes + src_shift
            dst = dst_nodes + dst_shift
            data = np.ones(len(src))
            adjusted_adj = sp.coo_matrix(
                (data, (src, dst)), shape=(total_num_nodes, total_num_nodes)
            )
            adjM_list.append(adjusted_adj)

        # Sum all adjusted adjacency matrices
        adjM = sum(adjM_list)
        adjM.setdiag(0) 
        adjM.eliminate_zeros()  # Ensure the sparsity of the matrix

        primary_shift = nodes_shift[self.primary_type]
        num_primary = node_counts[self.primary_type]
        primary_global_ids = np.arange(primary_shift, primary_shift + num_primary) 

        # Initialize labels_array as an array of length equal to the number of primary nodes
        labels_array = -1 * np.ones(num_primary, dtype=int)
        labels_array[:] = labels.cpu().numpy()

        # Adjust labels_array to be a torch tensor
        labels_tensor = torch.LongTensor(labels_array).to(self.device)

        # Step 7: Create edge2type mapping
        edge2type = {}
        edge_type_ids = {etype: idx for idx, etype in enumerate(edge_types)} 

        for etype in edge_types:
            src_nodes, dst_nodes = G.edges(etype=etype) 
            src_nodes = src_nodes.numpy()
            dst_nodes = dst_nodes.numpy()
            src_node_type, dst_node_type = edge_type_to_nodes[etype]
            src_shift = nodes_shift[src_node_type]
            dst_shift = nodes_shift[dst_node_type]

            src = src_nodes + src_shift
            dst = dst_nodes + dst_shift 
            for u, v in zip(src, dst):
                edge2type[(u, v)] = edge_type_ids[etype] 

        # Add self-loop edge type
        edge_type_self_loop = len(edge_types)
        for i in range(total_num_nodes):
            if (i, i) not in edge2type:
                edge2type[(i, i)] = edge_type_self_loop

        edge_type_offset = len(edge_types) + 1
        for etype in edge_types:
            src_nodes, dst_nodes = G.edges(etype=etype)
            src_nodes = src_nodes.numpy()
            dst_nodes = dst_nodes.numpy()
            src_node_type, dst_node_type = edge_type_to_nodes[etype]
            src_shift = nodes_shift[src_node_type]
            dst_shift = nodes_shift[dst_node_type]

            src = src_nodes + src_shift
            dst = dst_nodes + dst_shift
            for u, v in zip(src, dst):
                if (v, u) not in edge2type:
                    edge2type[(v, u)] = edge_type_ids[etype] + edge_type_offset

        adjM = adjM.tocsr()
        g = dgl.from_scipy(adjM)
        g = dgl.remove_self_loop(g)
        g = dgl.add_self_loop(g)

        edges = g.edges()
        u_list = edges[0].tolist()
        v_list = edges[1].tolist()
        e_feat = []
        for u, v in zip(u_list, v_list):
            e_feat.append(edge2type.get((u, v)))
        e_feat = torch.tensor(e_feat, dtype=torch.long).to(self.device)

        num_etypes = max(edge2type.values()) + 1
        in_dims = [features[node_type].shape[1] for node_type in node_types]
        num_classes = int(max(labels)) + 1

        g = g.to(self.device)

        print(g)

        return (
            g,
            features_list,
            labels_tensor,
            e_feat,
            num_etypes,
            in_dims,
            num_classes,
            primary_shift,
            primary_global_ids,
        )

    def create_model(self):
        args = self.args
        heads = [args.SimpleHGN_num_heads] * args.SimpleHGN_num_layers + [1]
        self.model = myGAT(
            self.g,
            args.SimpleHGN_edge_feats,
            self.num_etypes,
            self.in_dims,
            args.SimpleHGN_hidden_dim,
            self.num_classes,
            args.SimpleHGN_num_layers,
            heads,
            F.elu,
            args.SimpleHGN_dropout,
            args.SimpleHGN_dropout,
            args.SimpleHGN_slope,
            True,
            0.05,
            self.device,
        ).to(self.device)

    def predict(self, G, eval=False):

        return self.model(self.features_list, self.e_feat)

    def get_attention(self):
        return self.model.attentions

    def get_gradient(self, two_hop_neighbors_by_type):
        for feat in self.features_list:
            feat.requires_grad = True

        self.model.zero_grad()
        for feat in self.features_list:
            if feat.grad is not None:
                feat.grad.zero_()

        logits = self.model(self.features_list, self.e_feat)
        loss = F.cross_entropy(
            logits[self.train_idx_global],
            self.labels_tensor[self.train_idx_global - self.primary_shift],
        )

        loss.backward()
        gradients = [feat.grad.clone() for feat in self.features_list]
        gradients = torch.cat(gradients, dim=0)

        for feat in self.features_list:
            feat.requires_grad = False

        corresponding_gradients = {}
        for type in two_hop_neighbors_by_type.keys():
            nodes = list(two_hop_neighbors_by_type[type])
            node_index = list(map(lambda x: x + self.node_shift[type], nodes))
            corresponding_gradients[type] = gradients[node_index]
        return corresponding_gradients

    def evaluate(self, G):
        self.model.eval()
        features = {ntype: G.nodes[ntype].data['inp'] for ntype in G.ntypes}
        (
            g,
            features_list,
            labels_tensor,
            e_feat,
            num_etypes,
            in_dims,
            num_classes,
            primary_shift,
            primary_global_ids,
        ) = self.process_graph(G, features, self.labels)
        self.model.g = g
        logits = self.model(features_list,e_feat)
        return logits[primary_global_ids]

def get_model_class(model_name):
    if model_name == 'HGT':
        return HGTModel
    elif model_name == 'RGCN':
        return RGCNModel
    elif model_name == 'HAN':
        return HANModel
    elif model_name == "SimpleHGN":
        return SimpleHGNModel
    else:
        raise NotImplementedError(f"Model {model_name} not implemented")


def create_victim_model(model_name, args, hg, text_attribute, labels, num_classes, primary_type, metapaths,
                        train_idx, val_idx, test_idx):
    model_class = get_model_class(model_name)
    model = model_class(args, hg, text_attribute, labels, num_classes, primary_type, metapaths,
                        train_idx, val_idx, test_idx)
    return model
