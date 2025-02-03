import argparse
from data.utils import *
from utils import *
from backdoor import construct_backdoor_model, calculate_stealthiness_score
import pickle
import os
import json
import datetime
from copy import deepcopy
from evaluation import evaluation
from victim_models.victim_models import *


os.environ["DGLBACKEND"] = "pytorch"
os.environ["DGL_NUM_WORKER_THREADS"] = "1"

import dgl
#--------debug import--------
from test_func import *
#--------debug import--------
#-------------------------------------------------General Parameters-------------------------------------------------------
parser = argparse.ArgumentParser(description="Parameters")
parser.add_argument("--random_seed", type = int, help = "random seed", default = 999)
parser.add_argument("--dataset", help="dataset choice", default="IMDB", choices=["ACM","DBLP","IMDB"])
parser.add_argument("--model_name", help="model_name choice", default="HAN", choices=["HGT",'HAN','SimpleHGN'])
parser.add_argument("--target_class", type = int, default = 2, help="The target class of backdoor attack")
parser.add_argument("--chosen_poison_nodes_method", type = str, default = 'degree', help='the method of choosing poison node, backup: [degree, random, pagerank, cluster]')
parser.add_argument("--chosen_influential_nodes_method", type = str, default = 'cluster', help = 'the method of choosing influential node which connect to trigger nodes, backup: [degree, llm, random, pagerank,cluster]')
parser.add_argument("--trigger_type", type = str, default = 'actor', help = 'the trigger type') # 后期 可能会改变成自动选择 现在先是手动选择
parser.add_argument("--poison_set_ratio", type = float, default = 0.1)
parser.add_argument("--training", type = bool, default = True, help = "whether in a train phase")
parser.add_argument("--backdoor_type", type = str, default = 'HeteroBA', help='the backdoor method, backup: [HeteroBA, UGBA, CGBA]')
#------------------------------------------------------UGBA Parameter--------------------------------------------------------------------
parser.add_argument("--UGBA_hidden", type = int, default = 64, help='UGBA trojan net hidden layer size')
parser.add_argument("--UGBA_lr" , type = float, default = 0.001, help = 'UGBA learning rate')
parser.add_argument("--UGBA_trigger_size" , type = int, default = 3, help = 'UGBA trigger size')
parser.add_argument("--UGBA_trojan_epochs" , type = int, default = 100, help = 'UGBA trojan training epochs')
parser.add_argument("--UGBA_inner" , type = int, default = 1, help = 'UGBA inner training epochs')
parser.add_argument("--UGBA_thrd" , type = float, default = 0.5, help = 'UGBA thrshold')
parser.add_argument("--UGBA_target_loss_weight" , type = float, default = 1, help = 'UGBA target loss weight')
parser.add_argument("--UGBA_homo_loss_weight" , type = float, default = 100, help = 'UGBA homo loss weight')
parser.add_argument("--UGBA_homo_boost_thrd" , type = float, default = 0.5, help = 'UGBA homo boost threshold')
#----------------------------------------------------CGBA Parameter----------------------------------------------------------------------------
parser.add_argument("--CGBA_trigger_size", type = int, default = 1, help = 'CGBA trigger size')

#-------------------------------------------------Trainset Poison Chosen Parameters-------------------------------------------------------
parser.add_argument("--cluster_K", type = int, default = 2, help = "K Value of K means in cluser")
parser.add_argument("--dis_weight", type = float, default = 0.5, help = "K Value of K means in cluser")

#---------------------------------------------Model Parameters-------------------------------------------------------
parser.add_argument("--HGT_n_hid",type = int, default = 64, help="Number of HGT hidden units")
parser.add_argument("--RGCN_n_hid",type = int, default = 128, help="Number of RGCN hidden units")
parser.add_argument("--HAN_n_hid", type = int, default = 64,  help="HAN hidden units")
parser.add_argument("--HAN_num_heads",type = list, default = [4,4], help="HAN number of heads")
parser.add_argument("--HAN_dropout", type = float, default = 0.2, help="HAN dropout rate")
parser.add_argument("--SimpleHGN_hidden_dim", type=int, default=64, help='SimpleHGN Dimension of the node hidden state. Default is 64.')
parser.add_argument("--SimpleHGN_num_heads",type=int, default=8, help='SimpleHGN Number of the attention heads. Default is 8.')
parser.add_argument("--SimpleHGN_num_layers",type=int, default=4)
parser.add_argument("--SimpleHGN_dropout", type=float, default = 0.1, help = "SimpleHGN dropout rate")
parser.add_argument('--SimpleHGN_edge_feats', type=int, default=64, help = "SimpleHGN edge features")
parser.add_argument('--SimpleHGN_slope', type=float, default=0.05, help = "SimpleHGN slope")
parser.add_argument('--SimpleHGN_patience', type=int, default=100, help = "trianing SimpleHGN patience ")
parser.add_argument('--SimpleHGN_lr', type=float, default=5e-4, help = "SimpleHGN learning rate")
parser.add_argument('--SimpleHGN_weight_decay', type=float, default=1e-4)
parser.add_argument("--clip", type = float, default = 1.0, help='Gradient clipping')
parser.add_argument("--max_lr", type = float, default = 1e-3, help='Maximum learning rate')
parser.add_argument("--n_epoch", type = int, default = 400, help="Number of epochs")
parser.add_argument("--device", help="device", default="cuda")

#---------------------------------------------Model Parameters-------------------------------------------------------

args = parser.parse_args()
dataset = args.dataset
model_name = args.model_name
device = args.device
random_seed = args.random_seed


def set_seed(random_seed):
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # 强制 GPU 操作同步执行
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("high")
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    dgl.seed(random_seed)
    dgl.random.seed(random_seed)

def load_dataset_preprocess(dataset_name):
    dataset_name_lower = dataset_name.lower()
    if dataset_name_lower == 'ACM':
        from data.ACM.acm_preprocess import load_acm_raw
        return load_acm_raw()
    elif dataset_name_lower == 'IMDB':
        from data.IMDB.imdb_preprocess import load_imdb_raw
        return load_imdb_raw()
    elif dataset_name_lower == 'DBLP':
        from data.DBLP.dblp_preprocess import load_dblp_raw
        return load_dblp_raw()
    else:
        raise ValueError(f"Unknown dataset name：{dataset_name}")

def load_dataset(dataset_name):
    file = f"./data/{args.dataset}/{args.dataset}_data.pkl"
    if os.path.exists(file):
        with open(file, 'rb') as file:
            data = pickle.load(file)
            hg, hete_adjs, features, text_attribute, labels, num_classes, primary_type, metapaths, \
                train_idx, val_idx, test_idx = (
                data['hg'],
                data['hete_adjs'],
                data['features'],
                data['text_attribute'],
                data['labels'],
                data['num_classes'],
                data['primary_type'],
                data['metapaths'],
                data['train_idx'],
                data['val_idx'],
                data['test_idx']
            )
    else:
        (hg, hete_adjs, features, text_attribute, labels, num_classes, primary_type, metapaths,
        train_idx, val_idx, test_idx) = load_dataset_preprocess(dataset_name)
    edge_template = load_edge_templates(dataset_name)
    hg = update_hg_features(hg, args, features)
    posion_trainset_index, posion_testset_index, clean_trainset_index, clean_testset_index, trainset_index, testset_index, validation_set_index, poison_labels, homo_g, node_mapping = split_index(
        hg=hg,
        primary_type=primary_type,
        labels=deepcopy(labels),
        text_attribute=text_attribute,
        args=args,
        # 其他参数作为kwargs传入
        hete_adjs=hete_adjs,
        features=features,
        num_classes=num_classes,
        metapaths=metapaths,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        backdoor_type = args.backdoor_type,
    )
    return hg, hete_adjs,text_attribute, edge_template, features, labels, poison_labels, num_classes, posion_trainset_index, posion_testset_index,\
           clean_trainset_index, clean_testset_index, trainset_index, testset_index, validation_set_index, primary_type, metapaths, homo_g, node_mapping


def save_trainset(random_seed, posion_trainset_index, posion_testset_index, accuracy, CAD, ASR):
    # Convert NumPy arrays or PyTorch tensors to Python lists
    posion_trainset_index = posion_trainset_index.tolist() if hasattr(posion_trainset_index, "tolist") else posion_trainset_index
    posion_testset_index = posion_testset_index.tolist() if hasattr(posion_testset_index, "tolist") else posion_testset_index
    # wrong_indices = wrong_indices.tolist() if hasattr(wrong_indices, "tolist") else wrong_indices

    # Helper function to convert NumPy data types to native Python types
    def convert_to_builtin(o):
        if isinstance(o, np.generic):
            return o.item()
        elif isinstance(o, list):
            return [convert_to_builtin(element) for element in o]
        elif isinstance(o, dict):
            return {convert_to_builtin(k): convert_to_builtin(v) for k, v in o.items()}
        else:
            return o

    data = {
        "random_seed": int(random_seed),
        "posion_trainset_index": convert_to_builtin(posion_trainset_index),
        "posion_testset_index": convert_to_builtin(posion_testset_index),
        "poison_wrong_test_indices": convert_to_builtin(accuracy),
        "CAD": float(CAD),
        "ASR": float(ASR)
    }

    os.makedirs("randomseed_trainset", exist_ok=True)
    file = f"{random_seed}.json"
    with open(f"randomseed_trainset/{file}", 'w') as f:
        json.dump(data, f, indent=4)


def save_experiment_results(args, CAD, ASR, accuracy, stealthiness_score, save_dir="experiment_results"):
    """
    保存实验结果到文本文件中。

    参数:
        args: 参数对象，包含实验配置
        CAD: Clean Accuracy Drop
        ASR: Attack Success Rate
        accuracy: 模型准确率
        save_dir: 保存结果的目录路径
    """
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)

    # 构建文件名，包含关键参数
    filepath = os.path.join(save_dir, "experiment_results_CK.txt")
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # 格式化时间戳

    # 构建要写入的内容
    content = f""" {"=" * 50}
    Timestamp: {timestamp}

    Configuration Parameters:
    Dataset: {args.dataset}
    Victim Model: {args.model_name}
    Backdoor Type: {args.backdoor_type}
    Target Class: {args.target_class}
    Trigger Type: {args.trigger_type}
    Poison Set Ratio: {args.poison_set_ratio}
    Random Seed: {args.random_seed}

    Performance Metrics:
    Clean Accuracy Drop (CAD): {CAD:.4f}
    Attack Success Rate (ASR): {ASR:.4f}
    Model Accuracy: {accuracy:.4f}
    Stealthiness Score: {stealthiness_score:.4f}

    {"=" * 50}
    
    """

    # 写入文件
    with open(filepath, 'a') as f:
        f.write(content)

    print(f"Results have been saved to: {filepath}")


if __name__ == '__main__':
    set_seed(random_seed)

    hg, hete_adjs, text_attribute, edge_template, features, clean_labels, poison_labels, num_classes, posion_trainset_index, posion_testset_index, \
        clean_trainset_index, clean_testset_index, trainset_index, testset_index, validation_set_index, primary_type, metapaths, homo_g, node_mapping = load_dataset(dataset)
    backdoor_model = construct_backdoor_model(args, features, device, hg, hete_adjs, clean_labels, poison_labels, num_classes, posion_trainset_index, posion_testset_index, \
        clean_trainset_index, clean_testset_index, trainset_index, testset_index, primary_type, metapaths, text_attribute, edge_template, homo_g, node_mapping)
    poison_hg = backdoor_model.construct_posion_graph()

    victim_backdoor_model = create_victim_model(model_name, args, poison_hg.clone(), text_attribute, poison_labels, num_classes, primary_type, metapaths, \
                 trainset_index, validation_set_index, testset_index)
    victim_backdoor_model.train_victim_model()
    victim_clean_model = create_victim_model(model_name, args, copy.deepcopy(hg), text_attribute, clean_labels, num_classes, primary_type,
                              metapaths, trainset_index, validation_set_index, testset_index)
    victim_clean_model.train_victim_model()
    poison_test_hg = backdoor_model.construct_posion_graph(training=False)
    # ======================================================Claculate Stealthiness Score=============================================
    stealthiness_score, details = calculate_stealthiness_score(hg, poison_test_hg, args.trigger_type, clean_labels,
                                                      clean_trainset_index + clean_testset_index, args.target_class,
                                                      args.backdoor_type, posion_testset_index, primary_type)
    # print("stalthiness score is ", stealthiness_score)
    # ====================================================================End========================================================

    CAD, ASR, accuracy = evaluation(model_name, victim_backdoor_model, victim_clean_model, posion_testset_index, clean_testset_index,
               poison_labels, clean_labels, poison_test_hg, hg, primary_type, testset_index)
    #======================================debug info================================
    # save_trainset(random_seed, posion_trainset_index, posion_testset_index, accuracy, CAD, ASR)
    save_experiment_results(args, CAD, ASR, accuracy, stealthiness_score)
    # ======================================debug info================================


''' ACM-author as trigger
-------------------class = 2---------------------------
                   CAD            ASR     
HGT                0.0215        0.9950
SimpleHGN          0.0099        1.0000
HAN                0.0017        1.0000

DBLP-paper as trigger
-------------------class = 2---------------------------
                   CAD            ASR     
HGT                0.0230        0.9950
SimpleHGN          0.0036        1.0000
HAN                0.0099        1.0000


IMDB-actor as trigger
-------------------class = 2---------------------------
                   CAD            ASR     
HGT                0.0218        0.5140
SimpleHGN          0.0033        0.6963
HAN                0.0062        0.5093

'''