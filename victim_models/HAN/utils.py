from sklearn.metrics import f1_score
import datetime
import dgl
import errno
import numpy as np
import os
import pickle
import random
import torch
from scipy.sparse import csc_matrix

from dgl.data.utils import download, get_download_dir, _get_dgl_url
from pprint import pprint
from scipy import sparse
from scipy import io as sio
from torch_geometric.datasets import DBLP
import torch
import config_IMDB
from sklearn.model_selection import train_test_split
from torch_geometric.datasets import AMiner
from utils import *


def score(logits, labels):
    _, indices = torch.max(logits, dim=1)
    prediction = indices.long().cpu().numpy()
    labels = labels.cpu().numpy()
    accuracy = (prediction == labels).sum() / len(prediction)
    micro_f1 = f1_score(labels, prediction, average='micro')
    macro_f1 = f1_score(labels, prediction, average='macro')
    return accuracy, micro_f1, macro_f1

def score_detail(logits, labels):
    _, indices = torch.max(logits, dim=1)
    prediction = indices.long().cpu().numpy()
    labels = labels.cpu().numpy()
    acc_detail = np.array(prediction == labels,dtype='int')
    accuracy = (prediction == labels).sum() / len(prediction)
    micro_f1 = f1_score(labels, prediction, average='micro')
    macro_f1 = f1_score(labels, prediction, average='macro')
    return accuracy, micro_f1, macro_f1,acc_detail

def evaluate(model, g, features, labels, mask, loss_func, detail=False):
    model.eval()
    with torch.no_grad():
        logits = model(g, features)
    loss = loss_func(logits[mask], labels[mask])
    if detail:
        accuracy, micro_f1, macro_f1, acc_detail = score_detail(logits[mask], labels[mask])
        return acc_detail, accuracy, micro_f1, macro_f1
    else:
        accuracy, micro_f1, macro_f1 = score(logits[mask], labels[mask])
        return loss, accuracy, micro_f1, macro_f1

def reverse_string(s):
    return s[::-1]

def get_hg(dataname,given_adj_dict,target_type):
    if dataname == 'acm':
        hg_new = dgl.heterograph({
            ('paper', 'pa', 'author'): given_adj_dict['pa'].nonzero(),
            ('author', 'ap', 'paper'): given_adj_dict['ap'].nonzero(),
            ('paper', 'pf', 'field'): given_adj_dict['pf'].nonzero(),
            ('field', 'fp', 'paper'): given_adj_dict['fp'].nonzero(),
        })

    if dataname == 'aminer':
        hg_new = dgl.heterograph({
            ('paper', 'pa', 'author'): given_adj_dict['pa'].nonzero(),
            ('author', 'ap', 'paper'): given_adj_dict['ap'].nonzero(),
            ('paper', 'pr', 'ref'): given_adj_dict['pr'].nonzero(),
            ('ref', 'rp', 'paper'): given_adj_dict['rp'].nonzero(),
        })
    if dataname == 'dblp':
        hg_new = dgl.heterograph({
            ('paper', 'pa', 'author'): given_adj_dict['pa'].nonzero(),
            ('author', 'ap', 'paper'): given_adj_dict['ap'].nonzero(),
            ('paper', 'pt', 'term'): given_adj_dict['pt'].nonzero(),
            ('term', 'tp', 'paper'): given_adj_dict['tp'].nonzero()
        })
    if dataname == 'yelp':
        hg_new = dgl.heterograph({
            ('b', 'bu', 'u'): given_adj_dict['bu'].nonzero(),
            ('u', 'ub', 'b'): given_adj_dict['ub'].nonzero(),
            ('b', 'bs', 's'): given_adj_dict['bs'].nonzero(),
            ('s', 'sb', 'b'): given_adj_dict['sb'].nonzero(),
            ('b', 'bl', 'l'): given_adj_dict['bl'].nonzero(),
            ('l', 'lb', 'b'): given_adj_dict['lb'].nonzero(),
        })
    if dataname == 'imdb':
        hg_new = dgl.heterograph({
            ('m', 'ma', 'a'): given_adj_dict['ma'].nonzero(),
            ('a', 'am', 'm'): given_adj_dict['am'].nonzero(),
            ('m', 'md', 'd'): given_adj_dict['md'].nonzero(),
            ('d', 'dm', 'm'): given_adj_dict['dm'].nonzero(),
            ('m', 'mk', 'k'): given_adj_dict['mk'].nonzero(),
            ('k', 'km', 'm'): given_adj_dict['km'].nonzero(),
        })

        m_nodes = set(range(4275))  
        a_nodes = set(range(given_adj_dict['ma'].shape[1]))  
        d_nodes = set(range(given_adj_dict['md'].shape[1]))  
        k_nodes = set(range(given_adj_dict['mk'].shape[1]))

        connected_m_nodes = set(hg_new.nodes('m').numpy())
        connected_a_nodes = set(hg_new.nodes('a').numpy())
        connected_d_nodes = set(hg_new.nodes('d').numpy())
        connected_k_nodes = set(hg_new.nodes('k').numpy())

        isolated_m_nodes = m_nodes - connected_m_nodes
        isolated_a_nodes = a_nodes - connected_a_nodes
        isolated_d_nodes = d_nodes - connected_d_nodes
        isolated_k_nodes = k_nodes - connected_k_nodes

        hg_new.add_nodes(len(isolated_m_nodes), ntype='m')
        hg_new.add_nodes(len(isolated_a_nodes), ntype='a')
        hg_new.add_nodes(len(isolated_d_nodes), ntype='d')
        hg_new.add_nodes(len(isolated_k_nodes), ntype='k')
    embeddings_file = "embeddings/homo_" + dataname +"_embedding.pkl"
    hg_new ,edge_dict, node_dict= initialize_graph_data(hg_new, embeddings_file, target_type)
    return hg_new


def set_random_seed(seed=0):
    """Set random seed.
    Parameters
    ----------
    seed : int
        Random seed to use
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def mkdir_p(path, log=True):
    """Create a directory for the specified path.
    Parameters
    ----------
    path : str
        Path name
    log : bool
        Whether to print result for directory creation
    """
    try:
        os.makedirs(path)
        if log:
            print('Created directory {}'.format(path))
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path) and log:
            print('Directory {} already exists.'.format(path))
        else:
            raise

def get_date_postfix():
    """Get a date based postfix for directory name.
    Returns
    -------
    post_fix : str
    """
    dt = datetime.datetime.now()
    post_fix = '{}_{:02d}-{:02d}-{:02d}'.format(
        dt.date(), dt.hour, dt.minute, dt.second)

    return post_fix

def setup_log_dir(args, sampling=False):
    """Name and create directory for logging.
    Parameters
    ----------
    args : dict
        Configuration
    Returns
    -------
    log_dir : str
        Path for logging directory
    sampling : bool
        Whether we are using sampling based training
    """
    date_postfix = get_date_postfix()
    log_dir = os.path.join(
        args['log_dir'],
        '{}_{}'.format(args['dataset'], date_postfix))

    if sampling:
        log_dir = log_dir + '_sampling'

    mkdir_p(log_dir)
    return log_dir

# The configuration below is from the paper.
default_configure = {
    'lr': 0.005,             # Learning rate
    'num_heads': [8],        # Number of attention heads for node-level attention
    'hidden_units': 8,
    'dropout': 0.6,
    'weight_decay': 0.001,
    'num_epochs': 30,
    'patience': 20
}

sampling_configure = {
    'batch_size': 20
}

def setup(args):
    args.update(default_configure)
    set_random_seed(args['seed'])
    args['dataset'] = 'ACMRaw' if args['hetero'] else 'ACM'
    args['dataset'] = 'DBLPRaw' if args['hetero'] else 'DBLP'
    args['device'] = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    args['log_dir'] = setup_log_dir(args)
    return args

def setup_for_sampling(args):
    args.update(default_configure)
    args.update(sampling_configure)
    set_random_seed()
    args['device'] = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    args['log_dir'] = setup_log_dir(args, sampling=True)
    return args

def get_binary_mask(total_size, indices):
    mask = torch.zeros(total_size)
    mask[indices] = 1
    return mask.byte()

def load_acm(remove_self_loop):
    url = 'dataset/ACM3025.pkl'
    data_path = get_download_dir() + '/ACM3025.pkl'
    download(_get_dgl_url(url), path=data_path)

    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    labels, features = torch.from_numpy(data['label'].todense()).long(), \
                       torch.from_numpy(data['feature'].todense()).float()
    num_classes = labels.shape[1]
    labels = labels.nonzero()[:, 1]

    if remove_self_loop:
        num_nodes = data['label'].shape[0]
        data['PAP'] = sparse.csr_matrix(data['PAP'] - np.eye(num_nodes))
        data['PLP'] = sparse.csr_matrix(data['PLP'] - np.eye(num_nodes))

    # Adjacency matrices for meta path based neighbors
    # (Mufei): I verified both of them are binary adjacency matrices with self loops
    author_g = dgl.from_scipy(data['PAP'])
    subject_g = dgl.from_scipy(data['PLP'])
    gs = [author_g, subject_g]

    train_idx = torch.from_numpy(data['train_idx']).long().squeeze(0)
    val_idx = torch.from_numpy(data['val_idx']).long().squeeze(0)
    test_idx = torch.from_numpy(data['test_idx']).long().squeeze(0)

    num_nodes = author_g.number_of_nodes()
    train_mask = get_binary_mask(num_nodes, train_idx)
    val_mask = get_binary_mask(num_nodes, val_idx)
    test_mask = get_binary_mask(num_nodes, test_idx)

    print('dataset loaded')
    pprint({
        'dataset': 'ACM',
        'train': train_mask.sum().item() / num_nodes,
        'val': val_mask.sum().item() / num_nodes,
        'test': test_mask.sum().item() / num_nodes
    })

    return gs, features, labels, num_classes, train_idx, val_idx, test_idx, \
           train_mask, val_mask, test_mask


def load_acm_raw(remove_self_loop):
    assert not remove_self_loop
    set_random_seed(1)
    url = 'dataset/ACM.mat'
    #-------------------------server code-------------------------
    data_path = os.getcwd() + '/networks/dataset/ACM.mat'
    #-------------------------server code-------------------------
    data = sio.loadmat(data_path)
    p_vs_f = data['PvsL']
    p_vs_a = data['PvsA']
    p_vs_t = data['PvsT']
    p_vs_c = data['PvsC']

    # We assign
    # (1) KDD papers as class 0 (data mining),
    # (2) SIGMOD and VLDB papers as class 1 (database),
    # (3) SIGCOMM and MOBICOMM papers as class 2 (communication)
    conf_ids = [0, 1, 9, 10, 13]
    label_ids = [0, 1, 2, 2, 1]

    p_vs_c_filter = p_vs_c[:, conf_ids]
    p_selected = (p_vs_c_filter.sum(1) != 0).A1.nonzero()[0]
    p_vs_f = p_vs_f[p_selected]
    p_vs_a = p_vs_a[p_selected]
    p_vs_t = p_vs_t[p_selected]
    p_vs_c = p_vs_c[p_selected]#CSC
    hete_adjs = {'pa':p_vs_a, 'ap':p_vs_a.T,\
                 'pf':p_vs_f, 'fp':p_vs_f.T}

    hg = dgl.heterograph({
        ('paper', 'pa', 'author'): p_vs_a.nonzero(),
        ('author', 'ap', 'paper'): p_vs_a.transpose().nonzero(),
        ('paper', 'pf', 'field'): p_vs_f.nonzero(),
        ('field', 'fp', 'paper'): p_vs_f.transpose().nonzero(),
    })

    features = torch.FloatTensor(p_vs_t.toarray())

    pc_p, pc_c = p_vs_c.nonzero()
    labels = np.zeros(len(p_selected), dtype=np.int64)
    for conf_id, label_id in zip(conf_ids, label_ids):
        labels[pc_p[pc_c == conf_id]] = label_id
    labels = torch.LongTensor(labels)

    num_classes = 3

    float_mask = np.zeros(len(pc_p))
    for conf_id in conf_ids:
        pc_c_mask = (pc_c == conf_id)
        float_mask[pc_c_mask] = np.random.permutation(np.linspace(0, 1, pc_c_mask.sum()))
    train_idx = np.where(float_mask <= 0.2)[0]
    val_idx = np.where((float_mask > 0.2) & (float_mask <= 0.3))[0]
    test_idx = np.where(float_mask > 0.3)[0]

    num_nodes = hg.number_of_nodes('paper')
    train_mask = get_binary_mask(num_nodes, train_idx)
    val_mask = get_binary_mask(num_nodes, val_idx)
    test_mask = get_binary_mask(num_nodes, test_idx)

    def return_target_type():
        return "paper"

    def return_metapaths():
        return [['pa', 'ap'], ['pf', 'fp']]

    load_acm_raw.return_target_type = return_target_type
    load_acm_raw.return_metapaths = return_metapaths

    return hg, hete_adjs, features, labels, num_classes, train_idx, val_idx, test_idx, \
            train_mask, val_mask, test_mask

def edge_index_to_csc(edge_index: torch.Tensor, num_src_nodes: int = None, num_dst_nodes: int = None) -> csc_matrix:
    """
    Convert an edge index tensor to a CSC sparse matrix (adjacency matrix).

    Parameters:
    edge_index (torch.Tensor): The edge index tensor. Shape should be [2, E] where E is the number of edges.
    num_src_nodes (int, optional): The number of source nodes. If None, it will be inferred from the edge index.
    num_dst_nodes (int, optional): The number of destination nodes. If None, it will be inferred from the edge index.

    Returns:
    scipy.sparse.csc_matrix: The adjacency matrix in CSC format.
    """
    if num_src_nodes is None:
        num_src_nodes = edge_index[0, :].max().item() + 1
    if num_dst_nodes is None:
        num_dst_nodes = edge_index[1, :].max().item() + 1

    return csc_matrix((torch.ones(edge_index.shape[1], dtype=int), edge_index), shape=(num_src_nodes, num_dst_nodes))

def remove_zero_columns(sparse_matrix):
    # Find columns where all entries are zero
    non_zero_columns_mask = sparse_matrix.getnnz(axis=0) > 0
    # Remove columns where all entries are zero
    filtered_sparse_matrix = sparse_matrix[:, non_zero_columns_mask]
    return filtered_sparse_matrix


def remove_zero_rows(sparse_matrix):
    non_zero_counts = sparse_matrix.getnnz(axis=1)
    zero_row_indices = np.where(non_zero_counts == 0)[0]

    mask = np.ones(sparse_matrix.shape[0], dtype=bool)
    mask[zero_row_indices] = False
    filtered_sparse_matrix = sparse_matrix[mask]

    return filtered_sparse_matrix, zero_row_indices


def remove_specified_rows(sparse_matrix, row_indices):
    mask = np.ones(sparse_matrix.shape[0], dtype=bool)
    mask[row_indices] = False
    filtered_sparse_matrix = sparse_matrix[mask, :]

    return filtered_sparse_matrix


def load_dblp_raw(remove_self_loop):
    assert not remove_self_loop
    dataset = DBLP(root='/tmp/DBLP')
    data = dataset[0]

    p_vs_a = data['paper', 'to', 'author'].edge_index
    p_vs_t = data['paper', 'to', 'term'].edge_index
    p_vs_c = data['paper', 'to', 'conference'].edge_index
    p_vs_a = edge_index_to_csc(p_vs_a)
    p_vs_t = edge_index_to_csc(p_vs_t)
    p_vs_c = edge_index_to_csc(p_vs_c)


    conf_ids = [1, 5, 9]
    label_ids = [0, 1, 2]

    p_vs_c_filter = p_vs_c[:, conf_ids]
    p_selected = (p_vs_c_filter.sum(1) != 0).A1.nonzero()[0]
    p_vs_a = remove_zero_columns(p_vs_a[p_selected])
    p_vs_t = remove_zero_columns(p_vs_t[p_selected])
    p_vs_c = p_vs_c[p_selected]

    hete_adjs = {'pa':p_vs_a, 'ap':p_vs_a.T,\
                 'pt':p_vs_t, 'tp':p_vs_t.T}

    hg = dgl.heterograph({
        ('paper', 'pa', 'author'): p_vs_a.nonzero(),  
        ('author', 'ap', 'paper'): p_vs_a.transpose().nonzero(),
        ('paper', 'pt', 'term'): p_vs_t.nonzero(),
        ('term', 'tp', 'paper'): p_vs_t.transpose().nonzero()
    })
    features = torch.FloatTensor(p_vs_t.toarray())
    pc_p, pc_c = p_vs_c.nonzero() 
    labels = np.zeros(len(p_selected), dtype=np.int64)
    for conf_id, label_id in zip(conf_ids, label_ids):  
        labels[pc_p[pc_c == conf_id]] = label_id
    counts = np.bincount(labels)
    print(counts)
    labels = torch.LongTensor(labels)


    num_classes = len(label_ids)
    float_mask = np.zeros(len(pc_p))
    for conf_id in conf_ids:  
        pc_c_mask = (pc_c == conf_id)
        float_mask[pc_c_mask] = np.random.permutation(np.linspace(0, 1, pc_c_mask.sum())) 
    train_idx = np.where(float_mask <= 0.8)[0]
    val_idx = np.where((float_mask > 0.8) & (float_mask <= 0.9))[0]
    test_idx = np.where(float_mask > 0.9)[0]

    num_nodes = hg.number_of_nodes('paper')   
    train_mask = get_binary_mask(num_nodes, train_idx) 
    val_mask = get_binary_mask(num_nodes, val_idx)
    test_mask = get_binary_mask(num_nodes, test_idx)

    def return_target_type():
        return 'paper'

    def return_metapaths():
        return [['pa','ap'],['pt','tp']]

    load_dblp_raw.return_target_type = return_target_type
    load_dblp_raw.return_metapaths = return_metapaths

    return hg, hete_adjs, features, labels, num_classes, train_idx, val_idx, test_idx, \
            train_mask, val_mask, test_mask


def load_data(dataset, remove_self_loop=False):
    if dataset == 'ACM':
        return load_acm(remove_self_loop)
    elif dataset == 'ACMRaw':
        return load_acm_raw(remove_self_loop)
    else:
        return NotImplementedError('Unsupported dataset {}'.format(dataset))



def load_imdb_raw():
    data_config = config_IMDB.data_config
    data_path ='data/IMDB/IMDB.mat'
    data = sio.loadmat(data_path)
    m_vs_a = data['m_vs_a']  # paper-author
    m_vs_d = data['m_vs_d']  # paper-conference
    m_vs_k = data['m_vs_k']  # paper-term
    hete_adjs = {
        'ma': m_vs_a,
        'md': m_vs_d,
        'mk': m_vs_k,
        'am': m_vs_a.transpose(),
        'dm': m_vs_d.transpose(),
        'km': m_vs_k.transpose()
    }

    def scipy_to_edge_list(sp_matrix):
        coo = sp_matrix.tocoo()
        return (coo.row, coo.col)

    m_vs_a_edges = scipy_to_edge_list(m_vs_a)
    a_vs_m_edges = scipy_to_edge_list(m_vs_a.transpose())
    m_vs_d_edges = scipy_to_edge_list(m_vs_d)
    d_vs_m_edges = scipy_to_edge_list(m_vs_d.transpose())
    m_vs_k_edges = scipy_to_edge_list(m_vs_k)
    k_vs_m_edges = scipy_to_edge_list(m_vs_k.transpose())

    hg = dgl.heterograph({
        ('m', 'ma', 'a'): m_vs_a_edges,
        ('a', 'am', 'm'): a_vs_m_edges,
        ('m', 'md', 'd'): m_vs_d_edges,
        ('d', 'dm', 'm'): d_vs_m_edges,
        ('m', 'mk', 'k'): m_vs_k_edges,
        ('k', 'km', 'm'): k_vs_m_edges,
    })

    task_path = os.path.join(data_config['data_path'], data_config['dataset'], 'CF')
    if not os.path.exists(task_path):
        os.mkdir(task_path)
    task_data_path = os.path.join(data_config['data_path'], data_config['dataset'], 'CF',
                                  'data_test_{}.pkl'.format(data_config['test_ratio']))

    if data_config['resample'] or not os.path.exists(task_data_path):
        movie_label_path = os.path.join(os.path.dirname(data_path), 'index_label.txt')
        num_classes = 3
        data_list = []
        labels = []
        with open(movie_label_path) as f:
            for l in f.readlines():
                l = l.strip()
                idx, label = l.split(",")
                data_list.append(int(idx))
                labels.append(int(label) - 1)  # 0, 1, 2
        data_list = np.array(data_list)
        labels = np.array(labels)
        train_idx, test_idx = train_test_split(data_list, test_size=data_config['test_ratio'],
                                               random_state=data_config['random_seed'])
        features = data['m_feature'].toarray()
        with open(task_data_path, 'wb') as f:
            pickle.dump([features, labels, num_classes, train_idx, test_idx], f)
    else:
        with open(task_data_path, 'rb') as f:
            features, labels, num_classes, train_idx, test_idx = pickle.load(f)

    val_size = int(len(test_idx) * 0.2)
    val_idx = test_idx[:val_size]
    test_idx = test_idx[val_size:]

    train_mask = np.zeros(labels.shape[0], dtype=bool)
    val_mask = np.zeros(labels.shape[0], dtype=bool)
    test_mask = np.zeros(labels.shape[0], dtype=bool)

    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    features = torch.tensor(features, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long)
    train_mask = torch.tensor(train_mask, dtype=torch.bool)
    val_mask = torch.tensor(val_mask, dtype=torch.bool)
    test_mask = torch.tensor(test_mask, dtype=torch.bool)

    def return_target_type():
        return 'm'

    def return_metapaths():
        return [['ma','am'],['md','dm'],['mk','km']]

    load_imdb_raw.return_target_type = return_target_type
    load_imdb_raw.return_metapaths = return_metapaths

    return hg, hete_adjs, features, labels, num_classes, train_idx, val_idx, test_idx, train_mask, val_mask, test_mask


class EarlyStopping(object):
    def __init__(self, patience=10):
        dt = datetime.datetime.now()
        self.filename = 'early_stop_{}_{:02d}-{:02d}-{:02d}.pth'.format(
            dt.date(), dt.hour, dt.minute, dt.second)
        self.patience = patience
        self.counter = 0
        self.best_acc = None
        self.best_loss = None
        self.early_stop = False

    def step(self, loss, acc, model):
        if self.best_loss is None:
            self.best_acc = acc
            self.best_loss = loss
            self.save_checkpoint(model)
        elif (loss > self.best_loss) and (acc < self.best_acc):
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if (loss <= self.best_loss) and (acc >= self.best_acc):
                self.save_checkpoint(model)
            self.best_loss = np.min((loss, self.best_loss))
            self.best_acc = np.max((acc, self.best_acc))
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, model):
        """Saves model_name when validation loss decreases."""
        torch.save(model, self.filename)

    def load_checkpoint(self, model):
        """Load the latest checkpoint."""
        model.load_state_dict(torch.load(self.filename).state_dict())


def calculate_betweenness(nx_graph, node_mapping):
    import networkx as nx

    betweenness_centrality = nx.betweenness_centrality(nx_graph)
    node_betweenness = {}
    reversed_node_mapping = {}
    for key, value in node_mapping.items():
        if isinstance(value, (list, dict)):
            value = str(value)
        reversed_node_mapping[value] = key
    for k in betweenness_centrality.keys():
        node_betweenness[str(reversed_node_mapping[k])] = betweenness_centrality[k]

    edge_betweenness_file = {}
    edge_betweenness = nx.edge_betweenness_centrality(nx_graph)
    for k in edge_betweenness.keys():
        edge_betweenness_file[str((reversed_node_mapping[k[0]],reversed_node_mapping[k[1]]))] = edge_betweenness[k]

