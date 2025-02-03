import pickle
import networkx as nx
import copy
import numpy as np
from torch.cuda.amp import GradScaler

from victim_models.HGT.HGTModel import *
from victim_models.RGCN.RGCN import RGCN
from victim_models.HAN.model import HAN

def load_embeddings_file(dataset):
    embedding_file = f"embeddings/homo_{dataset}_embedding.pkl"
    with open(embedding_file, 'rb') as f:
        return pickle.load(f)

def heterograph_to_networkx(g,target_type):
    nx_graph = nx.Graph()
    node_mapping = {}
    types = copy.deepcopy(g.ntypes)
    types.remove(target_type)
    types.insert(0, target_type)

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

def create_victim_model(dataset, model_name, device, hg, labels, primary_type, metapaths, args):
    if model_name == 'HGT' or model_name == 'RGCN' or model_name == 'HAN':
        node_dict = {}
        edge_dict = {}
        for ntype in hg.ntypes:
            node_dict[ntype] = len(node_dict)
        for etype in hg.etypes:
            edge_dict[etype] = len(edge_dict)
            hg.edges[etype].data["id"] = (
                torch.ones(hg.num_edges(etype), dtype=torch.long) * edge_dict[etype]
        )
        hg = hg.to(device)


        if model_name == 'HGT':
            model = HGT(
                hg,
                node_dict,
                edge_dict,
                n_inp=hg.nodes[primary_type].data["inp"].shape[1],
                n_hid=args.HGT_n_hid,
                n_out=labels.max().item() + 1,
                n_layers=8,
                n_heads=4,
                use_norm=True,
            ).to(device)
        elif model_name == 'RGCN':
            model = RGCN(in_feats = hg.nodes[primary_type].data["inp"].shape[1],
                              hidden_feats = args.RGCN_n_hid,
                              out_feats = labels.max().item() + 1,
                              rels = hg.etypes,
                              target_type = primary_type,
                              ).to(device)
        elif model_name == "HAN":
            model = HAN(meta_paths=metapaths,
                        in_size = hg.nodes[primary_type].data["inp"].shape[1],
                        hidden_size = args.HAN_n_hid,
                        out_size = labels.max().item() + 1,
                        num_heads = args.HAN_num_heads,
                        dropout=args.HAN_dropout
                        ).to(device)
        else:
            raise NotImplementedError(f"Model {model_name} not implemented")
        return model, hg


def train_victim_model(model, G, target_type, train_idx, test_idx, labels, device, args):
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, total_steps = args.n_epoch, max_lr = args.max_lr)
    scaler = GradScaler()


    best_val_acc = torch.tensor(0)
    best_test_acc = torch.tensor(0)
    best_model = None


    for epoch in np.arange(args.n_epoch) + 1:
        model.train()
        optimizer.zero_grad()


        if args.model_name == 'HGT':
            logits = model(G, target_type)
        elif args.model_name == 'RGCN':
            logits = model(G, G.ndata['inp'])
        elif args.model_name == "HAN":
            logits = model(G, G.ndata['inp'][target_type])
        else:
            raise TypeError("Did not implement the model_name")
        loss = F.cross_entropy(logits[train_idx], labels[train_idx].to(device))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        scheduler.step()
        if epoch % 5 == 0:
            model.eval()
            if args.model_name == 'HGT':
                logits = model(G, target_type)
            elif args.model_name == 'RGCN':
                logits = model(G, G.ndata['inp'])
            elif args.model_name == "HAN":
                logits = model(G, G.ndata['inp'][target_type])
            pred = logits.argmax(1).cpu()
            train_acc = (pred[train_idx] == labels[train_idx]).float().mean()
            test_acc = (pred[test_idx] == labels[test_idx]).float().mean()
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_model = copy.deepcopy(model)

            print(
                "Epoch: %d LR: %.5f Loss %.4f, Train Acc %.4f, Validation Acc %.4f (Best %.4f)"
                % (
                    epoch,
                    optimizer.param_groups[0]["lr"],
                    loss.item(),
                    train_acc.item(),
                    test_acc.item(),
                    best_test_acc.item(),
                )
            )
    return best_model