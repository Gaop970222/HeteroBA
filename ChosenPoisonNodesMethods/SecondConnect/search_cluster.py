from .search_base import SearchSecondConnectNode
from victim_models.victim_models import SimpleHGNModel
from .utils import plot_embedding
import torch

class SearchNodesByCluster(SearchSecondConnectNode):
    def __init__(self, the_method_search_influential_nodes, influential_numbers_by_type,
                 two_hop_neighbors_by_type, primary_type, hg, target_class, clean_labels,
                 text_attribute, edge_template, metapaths, trainset_index, testset_index, args):
        super().__init__(the_method_search_influential_nodes, influential_numbers_by_type,
                         two_hop_neighbors_by_type, primary_type, hg, target_class, clean_labels)
        self.candidate_nodes_by_type = {}
        self.initial_pool_by_type = {}  # 存储每种类型固定的 2n 个候选节点
        self.text_attribute = text_attribute
        self.edge_template = edge_template
        self.metapaths = metapaths
        self.trainset_index = trainset_index
        self.testset_index = testset_index
        self.args = args
        self.embeddings = self.prepare_candidate_nodes()


    def prepare_candidate_nodes(self):
        # 预处理候选节点并为每种类型选择固定的 2n 个节点
        num_classes = max(self.clean_labels).item() + 1
        model = SimpleHGNModel(self.args, self.hg, self.text_attribute, self.clean_labels, num_classes,
                       self.primary_type, self.metapaths, self.trainset_index, self.testset_index,
                       self.testset_index)
        model.train_victim_model()
        embeddings = model.get_specific_embeddings(self.two_hop_neighbors_by_type)
        #=================================debug info=========================
        # attention = get_node_attention(model, self.hg)

        # =================================debug info=========================
        return embeddings


    def compute_cosine_similarity_matrix(self, embeddings):
        norm = embeddings.norm(p=2, dim=1, keepdim=True) # p=2: L2 范数
        normalized_embeddings = embeddings / (norm + 1e-10)  # 防止除以零

        # 计算余弦相似性矩阵
        similarity_matrix = torch.mm(normalized_embeddings, normalized_embeddings.t())
        return similarity_matrix


    def get_top_n_nodes(self, embeddings, n):
        similarity_matrix = self.compute_cosine_similarity_matrix(embeddings)
        average_similarity = self.compute_average_similarity(similarity_matrix)

        # 获取前n个节点的索引
        top_n_values, top_n_indices = torch.topk(average_similarity, n, largest=True)
        return top_n_indices.tolist()


    def compute_average_similarity(self, similarity_matrix):
        similarity_matrix.fill_diagonal_(0) # 自己和自己的相似度肯定最高 变成0
        average_similarity = similarity_matrix.mean(dim=1)# 求均值
        return average_similarity


    def getFilterNodes(self):
        selected_nodes_by_type = {}
        for node_type in self.influential_numbers_by_type.keys():
            n = self.influential_numbers_by_type[node_type]
            corresponding_embeddings = self.embeddings[node_type]
            centers = self.get_top_n_nodes(corresponding_embeddings, n)
            plot_embedding(corresponding_embeddings, centers, dataset=self.args.dataset, chosen_nodes_method=self.args.chosen_influential_nodes_method ,node_type = node_type, save_fig=True)
            # plot_embedding(corresponding_embeddings, self.get_top_n_nodes(corresponding_embeddings, n), node_type)
            selected_nodes = self.get_top_n_nodes(corresponding_embeddings, n)
            selected_nodes_by_type[node_type] = [self.two_hop_neighbors_by_type[node_type][n] for n in selected_nodes]

        return selected_nodes_by_type


