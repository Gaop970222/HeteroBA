import torch
import numpy as np
from scipy.stats import wasserstein_distance
from sklearn.metrics.pairwise import cosine_similarity

def accuracy(output, labels):
    """Return accuracy of output compared to labels.
    Parameters
    ----------
    output : torch.Tensor
        output from model
    labels : torch.Tensor or numpy.array
        node labels
    Returns
    -------
    float
        accuracy
    """
    if not hasattr(labels, '__len__'):
        labels = [labels]
    if type(labels) is not torch.Tensor:
        labels = torch.LongTensor(labels)
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def get_relevant_etypes(graph, trigger_type):
    """
    Get edge types that are relevant to the trigger type.

    Parameters:
    - graph: The graph (original or poisoned)
    - trigger_type: Type of trigger nodes

    Returns:
    - List of relevant edge types
    """
    return [
        etype for etype in graph.etypes
        if etype.startswith(trigger_type[0]) or etype.endswith(trigger_type[0])
    ]


def calculate_stealthiness_score(original_graph, poisoned_graph, trigger_type, clean_labels, clean_set, target_class,
                                 backdoor_type, posion_testset_index, primary_type):
    """
    Calculate the stealthiness score for backdoor attacks on heterogeneous graphs.

    Parameters:
    - original_graph: Original heterogeneous graph
    - poisoned_graph: Poisoned heterogeneous graph
    - trigger_type: Type of trigger nodes
    - clean_labels: Original node labels (for primary_type nodes)
    - clean_set: Clean node set (indices of primary_type nodes)
    - target_class: Target class for the attack
    - primary_type: The primary node type in the heterogeneous graph

    Returns:
    - float: Stealthiness score (higher is better)
    - dict: Detailed scores and analysis
    """
    import torch
    import numpy as np
    from scipy.stats import wasserstein_distance

    def get_nodes_info():
        """Get clean nodes (target_class) and trigger nodes information"""
        # Get clean nodes of target class (primary_type中属于target_class的clean节点)
        clean_target_class_nodes = [
            node for node in clean_set
            if clean_labels[node] == target_class
        ]

        if not clean_target_class_nodes:
            raise ValueError(f"No clean nodes found for target class {target_class}")

        # Get trigger nodes
        original_nodes = set(original_graph.nodes(trigger_type).tolist())
        poisoned_nodes = set(poisoned_graph.nodes(trigger_type).tolist())
        trigger_node_ids = list(poisoned_nodes - original_nodes)

        if not trigger_node_ids:
            raise ValueError("No trigger nodes found in the poisoned graph")

        trigger_features = poisoned_graph.nodes[trigger_type].data['inp'][trigger_node_ids]

      
        all_trigger_type_nodes_original = original_graph.nodes(trigger_type).tolist()
        clean_trigger_nodes = all_trigger_type_nodes_original  
        clean_features = original_graph.nodes[trigger_type].data['inp'][clean_trigger_nodes]

        return clean_target_class_nodes, trigger_node_ids, trigger_features, clean_features, clean_trigger_nodes

    def calc_feature_similarity(trigger_features, clean_features):
        """Calculate feature distribution similarity using Wasserstein distance"""
        if isinstance(trigger_features, torch.Tensor):
            trigger_feats = trigger_features.cpu().numpy()
        else:
            trigger_feats = trigger_features

        if isinstance(clean_features, torch.Tensor):
            clean_feats = clean_features.cpu().numpy()
        else:
            clean_feats = clean_features

        # Calculate Wasserstein distance for each dimension
        distances = []
        for i in range(trigger_feats.shape[1]):
            dist = wasserstein_distance(trigger_feats[:, i], clean_feats[:, i])
            distances.append(dist)

        trigger_mean = np.mean(trigger_feats, axis=0)  # shape: (1000,)
        clean_mean = np.mean(clean_feats, axis=0)
        dist = np.linalg.norm(trigger_mean - clean_mean)

        avg_dist = np.mean(distances)
        feat_sim = 1 / (1 + avg_dist)

        details = {
            'per_dimension_distances': distances,
            'average_distance': avg_dist,
            'similarity_score': feat_sim
        }

        return feat_sim, details

    def calc_structural_diversity(trigger_node_ids):
        """
        Calculate a structural similarity score based on degree patterns.
        The idea:
        1. Compute average degree of trigger nodes in poisoned_graph.
        2. Randomly sample the same number of clean nodes (of the same trigger_type) from original_graph.
        3. Compute average degree of this clean sample.
        4. Compare the difference. Smaller difference => Higher stealthiness.
        """

        def get_node_degree(g, ntype, node_ids):
            degrees = []
            for nid in node_ids:
                deg = 0
                for (src_type, etype, dst_type) in g.canonical_etypes:
                    if src_type == ntype:
                        deg += g.out_degrees(nid, etype=etype)
                    if dst_type == ntype:
                        deg += g.in_degrees(nid, etype=etype)
                degrees.append(deg)
            return degrees

        trigger_degrees = get_node_degree(poisoned_graph, trigger_type, trigger_node_ids)
        avg_trigger_deg = np.mean(trigger_degrees)

        original_trigger_nodes = original_graph.nodes(trigger_type).tolist()
        clean_candidate_nodes = original_trigger_nodes

        if len(clean_candidate_nodes) < len(trigger_node_ids):
            clean_sample = np.random.choice(clean_candidate_nodes, size=len(trigger_node_ids), replace=True)
        else:
            clean_sample = np.random.choice(clean_candidate_nodes, size=len(trigger_node_ids), replace=False)

        clean_degrees = get_node_degree(original_graph, trigger_type, clean_sample)
        avg_clean_deg = np.mean(clean_degrees)

        degree_diff = abs(avg_trigger_deg - avg_clean_deg)
        struct_div = 1 / (1 + degree_diff)

        details = {
            'avg_trigger_degree': float(avg_trigger_deg),
            'avg_clean_degree': float(avg_clean_deg),
            'degree_diff': float(degree_diff),
            'struct_div_score': float(struct_div)
        }

        return struct_div, details
    if backdoor_type != "CGBA":
        clean_target_class_nodes, trigger_node_ids, trigger_features, clean_features, clean_trigger_nodes = get_nodes_info()

        # Weights for combination
        w1, w2 = 0.5, 0.5

        feat_sim, feat_details = calc_feature_similarity(trigger_features, clean_features)
        struct_div, struct_details = calc_structural_diversity(trigger_node_ids)

        # Calculate final score
        final_score = (w1 * feat_sim +
                       w2 * struct_div)

        # Compile detailed results
        details = {
            'feature_similarity': {
                'score': feat_sim,
                'weight': w1,
                'weighted_score': w1 * feat_sim,
                'details': feat_details
            },
            'structural_diversity': {
                'score': struct_div,
                'weight': w2,
                'weighted_score': w2 * struct_div,
                'details': struct_details
            },
            'final_score': final_score,
            'nodes_info': {
                'trigger_nodes_count': len(trigger_node_ids),
                'clean_target_class_nodes_count': len(clean_target_class_nodes)
            }
        }

        return final_score, details

    else:
        original_features = original_graph.nodes[primary_type].data['inp']
        poisoned_features = poisoned_graph.nodes[primary_type].data['inp']

        poison_node_ids = torch.as_tensor(posion_testset_index, dtype=torch.long)

        feat_clean = original_features[poison_node_ids]
        feat_poisoned = poisoned_features[poison_node_ids]

        c_np = feat_clean.numpy()
        p_np = feat_poisoned.numpy()

        per_dim_distances = []
        for dim_i in range(c_np.shape[1]):
            dist = wasserstein_distance(c_np[:, dim_i], p_np[:, dim_i])
            per_dim_distances.append(dist)
        avg_wasserstein = np.mean(per_dim_distances)
        feature_stealth_score = 1.0 / (1.0 + avg_wasserstein)

        c_all = feat_clean.cpu().numpy()
        p_all = feat_poisoned.cpu().numpy()
        l2_diff_all = np.linalg.norm(p_all - c_all, axis=1).mean()
        feature_stealth_score_all = 1.0 / (1.0 + l2_diff_all)




