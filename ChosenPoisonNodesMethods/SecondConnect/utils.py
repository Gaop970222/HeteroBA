import os
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import pandas as pd
import torch

def draw_long_tail_distribution_for_hetero(
        hetero_attention,
        tail_quantile=0.9,
        use_kde=True,
        log_scale=False,
        save_dir='attention_distribution'
):

    KLEIN_BLUE = "#002FA7"
    BRIGHT_YELLOW = "#FFD700"

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    sns.set_theme(
        style="white",
        rc={
            "axes.titleweight": "bold", 
            "axes.titlesize": 16,        
            "axes.labelsize": 14        
        }
    )

    node_types = sorted(hetero_attention.keys())

    for ntype in node_types:
        node_list = hetero_attention[ntype]
        scores_array = [score for (_, score) in node_list]

        if len(scores_array) == 0:
            print(f"[Warning]")
            continue

        scores_array = np.array(scores_array)
        threshold = np.quantile(scores_array, tail_quantile)

        scores_main = scores_array
        scores_tail = scores_array[scores_array >= threshold]

        fig_main, ax_main = plt.subplots(figsize=(7, 6))

        sns.histplot(
            x=scores_main,
            bins=50,
            color=KLEIN_BLUE,
            kde=False,
            stat="density",      
            ax=ax_main,
            alpha=0.7
        )
        if use_kde:
            sns.kdeplot(
                x=scores_main,
                color=BRIGHT_YELLOW,
                ax=ax_main,
                linewidth=4,
                cut=0 
            )

        ax_main.set_title(f"Overall Distribution of '{ntype}'",
                          fontsize=20, fontweight='bold')
        ax_main.set_xlabel("Attention Score", fontsize=18)
        ax_main.set_ylabel("Density" if use_kde else "Frequency", fontsize=18)
        if log_scale:
            ax_main.set_xscale("log")

        sns.despine(fig=fig_main, ax=ax_main)
        fig_main.tight_layout()

        if save_dir is not None:
            save_path_main = os.path.join(save_dir, f"{ntype}_main_distribution.png")
            fig_main.savefig(save_path_main, dpi=300)
            print(f"[Info] Saved figure (main) to {save_path_main}")

        plt.show()
        plt.close(fig_main)  

        fig_tail, ax_tail = plt.subplots(figsize=(7, 6))

        sns.histplot(
            x=scores_tail,
            bins=30,
            color=KLEIN_BLUE,
            kde=False,
            stat="density",
            ax=ax_tail,
            alpha=0.7
        )
        if use_kde:
            sns.kdeplot(
                x=scores_tail,
                color=BRIGHT_YELLOW,
                ax=ax_tail,
                linewidth=2,
                cut=0
            )

        ax_tail.set_title(f"Tail Distribution of '{ntype}' (score >= {threshold:.4f})",
                          fontsize=20, fontweight='bold')
        ax_tail.set_xlabel("Attention Score", fontsize=18)
        ax_tail.set_ylabel("Density" if use_kde else "Frequency", fontsize=18)
        if log_scale:
            ax_tail.set_xscale("log")

        if len(scores_tail) < 10:
            ax_tail.text(
                0.5, 0.9,
                "Few tail points",
                ha="center", va="center",
                transform=ax_tail.transAxes,
                color="red", fontsize=12
            )

        sns.despine(fig=fig_tail, ax=ax_tail)
        fig_tail.tight_layout()

        if save_dir is not None:
            save_path_tail = os.path.join(save_dir, f"{ntype}_tail_distribution.png")
            fig_tail.savefig(save_path_tail, dpi=300)
            print(f"[Info] Saved figure (tail) to {save_path_tail}")

        plt.show()
        plt.close(fig_tail)


def analysis_attention_degree(sorted_hetero, hg, target_class, labels,
                              primary_type, trigger_type, dataset, use_max=False, top_n=500):
    ntype_id_offset = {}
    current_offset = 0
    for ntype in hg.ntypes: 
        num_nodes_t = hg.num_nodes(ntype)
        ntype_id_offset[ntype] = (current_offset, current_offset + num_nodes_t)
        current_offset += num_nodes_t

    primary_labels = labels  
    target_local_ids = (primary_labels == target_class).nonzero(as_tuple=True)[0]
    A = set(target_local_ids.tolist())

    
    neighbor_to_As = defaultdict(set)

    for a_local_id in A:
        for etype in hg.canonical_etypes:
            src_ntype, rel, dst_ntype = etype
            if src_ntype == primary_type:
                srcs, dsts = hg.out_edges(a_local_id, etype=etype)
                for dst_local_id in dsts.tolist():
                    neighbor_to_As[(dst_ntype, dst_local_id)].add(a_local_id)

    result = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    for n1, a_set in neighbor_to_As.items():
        n1_type, n1_local_id = n1
        count_of_A = len(a_set) 
        for etype in hg.canonical_etypes:
            src_ntype, rel, dst_ntype = etype
            if src_ntype == n1_type:
                srcs, dsts = hg.out_edges(n1_local_id, etype=etype)
                for n2_local_id in dsts.tolist():
                    result[dst_ntype][n2_local_id][(n1_type, n1_local_id)] += count_of_A

    for ntype, ntype_dict in result.items():
        for local_id, local_dict in ntype_dict.items():
            grouped_by_type = defaultdict(list)
            for (nbr_type, nbr_local_id), cA in local_dict.items():
                grouped_by_type[nbr_type].append(cA)

            new_local_dict = {}
            for nbr_type, cA_list in grouped_by_type.items():
                if use_max:
                    new_local_dict[nbr_type] = max(cA_list)
                else:
                    new_local_dict[nbr_type] = sum(cA_list)

            result[ntype][local_id] = new_local_dict

    result = {k: dict(v) for k, v in result.items()}

    reordered_result = {}
    for key in result.keys():
        sorted_ids = [item[0] for item in sorted_hetero.get(key, [])]
        reordered_result[key] = {
            node_id: result[key][node_id] for node_id in sorted_ids
            if node_id in result[key]
        }

    def plot_neighbor_types_separately_for_key(
            reordered_result,
            sorted_hetero,
            main_key,
            dataset_name,  
            top_n=500
    ):


        sorted_pairs = sorted_hetero.get(main_key, [])[:top_n]
        node_ids = [item[0] for item in sorted_pairs]
        attention_values = [item[1] for item in sorted_pairs]

        neighbor_types = set()
        for nid in node_ids:
            if nid in reordered_result.get(main_key, {}):
                inner_dict = reordered_result[main_key][nid]
                neighbor_types.update(inner_dict.keys()) 

        for nbr_type in neighbor_types:
            values_for_this_type = []
            for nid in node_ids:
                val = 0
                if nid in reordered_result[main_key]:
                    val = reordered_result[main_key][nid].get(nbr_type, 0)
                values_for_this_type.append(val)

            min_len = min(len(attention_values), len(values_for_this_type))
            x = range(min_len)
            atn = attention_values[:min_len]
            nbr_vals = values_for_this_type[:min_len]

            fig, axes = plt.subplots(2, 1, figsize=(10, 8))

            axes[0].plot(x, atn, color='b', marker='o')
            axes[0].set_title(
                f"[{dataset_name}] [{main_key}] - Attention Curve (Top {min_len})\nNeighbor Type = {nbr_type}"
            )
            axes[0].set_xlabel("Index")
            axes[0].set_ylabel("Attention Value")
            axes[0].grid(True)

            axes[1].plot(x, nbr_vals, color='r', marker='x')
            axes[1].set_title(
                f"[{dataset_name}] [{main_key}] - Reordered Result (Neighbor Type = {nbr_type})"
            )
            axes[1].set_xlabel("Index")
            axes[1].set_ylabel("Aggregate Value")
            axes[1].grid(True)

            fig.tight_layout()

            file_name = f"{dataset_name}_{main_key}_{nbr_type}.png"
            plt.savefig(file_name, dpi=300, bbox_inches='tight')
            print(f"Image save to: {file_name}")

            plt.show()

    def plot_neighbor_types_separately_for_all_keys(
            reordered_result, sorted_hetero, top_n=500
    ):
        for main_key in reordered_result.keys():
            plot_neighbor_types_separately_for_key(
                reordered_result, sorted_hetero, main_key,dataset,top_n=top_n
            )

    plot_neighbor_types_separately_for_all_keys(reordered_result, sorted_hetero, top_n=500)

    return reordered_result


import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap


def plot_embedding(
        embedding,
        highlight_indices=None,
        labels=None,
        method='PCA',
        n_components=2,
        perplexity=30,
        n_iter=1000,
        random_state=42,
        palette="viridis",
        figsize=(12, 10), 
        highlight_color='#FFF700',
        highlight_size=100,
        highlight_marker='X',
        dataset=None,
        chosen_nodes_method=None,
        node_type=None,
        save_fig=False
):
    sns.set(style="white", context="notebook") 

    if isinstance(embedding, torch.Tensor):
        embedding = embedding.detach().cpu().numpy()
    if labels is not None and isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    if embedding.ndim != 2:
        raise ValueError("embedding should be 2d matrix with shape(N, D)")

    if method == 'PCA':
        reducer = PCA(n_components=n_components, random_state=random_state)
        embedding_reduced = reducer.fit_transform(embedding)
    elif method == 't-SNE':
        reducer = TSNE(n_components=n_components,
                       perplexity=perplexity,
                       n_iter=n_iter,
                       random_state=random_state)
        embedding_reduced = reducer.fit_transform(embedding)
    elif method == 'UMAP':
        reducer = umap.UMAP(n_components=n_components, random_state=random_state)
        embedding_reduced = reducer.fit_transform(embedding)
    else:
        raise ValueError("Unsupported method.")

    df = pd.DataFrame({
        f'{method}1': embedding_reduced[:, 0],
        f'{method}2': embedding_reduced[:, 1]
    })

    if labels is not None:
        df['Label'] = labels

    plt.figure(figsize=figsize)

    # 绘制散点图
    if labels is not None:
        unique_labels = np.unique(labels)
        num_labels = len(unique_labels)
        if isinstance(palette, str):
            palette = sns.color_palette(palette, n_colors=num_labels)
        scatter = sns.scatterplot(
            data=df,
            x=f'{method}1',
            y=f'{method}2',
            hue='Label',
            palette=palette,
            legend='full',
            alpha=0.7
        )
    else:
        scatter = sns.scatterplot(
            data=df,
            x=f'{method}1',
            y=f'{method}2',
            color='#002FA7',
            legend=False,
            alpha=0.7
        )

    # 高亮显示
    selected_indices = []
    if highlight_indices is not None and len(highlight_indices) > 0:
        highlight_indices = np.array(highlight_indices, dtype=int)
        highlight_coords = embedding_reduced[highlight_indices]
        plt.scatter(
            highlight_coords[:, 0],
            highlight_coords[:, 1],
            c=highlight_color,
            s=highlight_size,
            marker=highlight_marker,
            edgecolors='k',
            label='Highlighted Nodes'
        )
        selected_indices = highlight_indices.tolist()
        plt.legend()

    title_parts = []
    if dataset:
        title_parts.append(str(dataset))
    if node_type:
        title_parts.append(str(node_type))
    full_title = "Scatter Plot" + (" - " + "-".join(title_parts) if title_parts else "")
    plt.title(full_title, fontsize=32, fontweight='bold', pad=20)  
    
    plt.xlabel('X', fontsize=28, labelpad=10) 
    plt.ylabel('Y', fontsize=28, labelpad=10)

    plt.tick_params(axis='both', which='major', labelsize=12)  

    scatter.grid(False)

    plt.tight_layout()

    if save_fig:
        name_parts = []
        if dataset:
            name_parts.append(str(dataset))
        if chosen_nodes_method:
            name_parts.append(str(chosen_nodes_method))
        if node_type:
            name_parts.append(str(node_type))
        if name_parts:
            save_path = "-".join(name_parts) + ".png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Image save to {save_path}")
        else:
            print("don't point dataset/chosen_nodes_method/node_type")

    plt.show()
