from .search_cluster import SearchNodesByCluster
from .search_degree import SearchNodesByDegree
from .search_random import SearchNodesByRandom
from .search_pagerank import SearchNodesByPageRank
from .search_llm import SearchNodesByLLM
from .search_gradient import SearchNodesByGradient
from .search_attention import SearchNodesByAttention
from .search_filter_attention import SearchNodesByFilteredAttention
from .search_by_kmeans import SearchNodesByKMeans


def create_victim_model(the_method_search_influential_nodes, influential_numbers_by_type,
                        two_hop_neighbors_by_type, primary_type, hg, target_class, clean_labels,
                        metapaths, trainset_index, testset_index, args, text_attribute=None, edge_template = None):
    method_map = {
        "degree": SearchNodesByDegree,
        "random": SearchNodesByRandom,
        "cluster": SearchNodesByCluster,
        "attention": SearchNodesByAttention,
    }

    if the_method_search_influential_nodes not in method_map:
        raise NotImplementedError("This method has not been implemented")

    method_class = method_map[the_method_search_influential_nodes]

    if the_method_search_influential_nodes == "llm":
        if text_attribute is None:
            raise ValueError("LLM search method requires text_attribute parameter")
        return method_class(the_method_search_influential_nodes, influential_numbers_by_type,
                            two_hop_neighbors_by_type, primary_type, hg, target_class,
                            clean_labels, text_attribute, edge_template)
    elif ("cluster" in the_method_search_influential_nodes or
          the_method_search_influential_nodes == 'attention' ):
        return method_class(the_method_search_influential_nodes, influential_numbers_by_type,
                            two_hop_neighbors_by_type, primary_type, hg, target_class,
                            clean_labels, text_attribute, edge_template, metapaths,
                            trainset_index, testset_index, args)

    return method_class(the_method_search_influential_nodes, influential_numbers_by_type,
                        two_hop_neighbors_by_type, primary_type, hg, target_class,
                        clean_labels)