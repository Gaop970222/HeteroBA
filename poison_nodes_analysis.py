import os
import pickle
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def analyze_poison_nodes(homo_g, poison_nodes, labels, target_class, label):
    centrality = calculate_centrality_measures(homo_g)

    centrality_poison_nodes = {node: {'betweenness': centrality['betweenness'].get(node, 0),
                                      'closeness': centrality['closeness'].get(node, 0),
                                      'eigenvector': centrality['eigenvector'].get(node, 0)}
                               for node in poison_nodes}

    avg_betweenness = np.mean([v['betweenness'] for v in centrality_poison_nodes.values()])
    avg_closeness = np.mean([v['closeness'] for v in centrality_poison_nodes.values()])
    avg_eigenvector = np.mean([v['eigenvector'] for v in centrality_poison_nodes.values()])
    print(f"{label} Poison Nodes - Average Betweenness Centrality: {avg_betweenness}")
    print(f"{label} Poison Nodes - Average Closeness Centrality: {avg_closeness}")
    print(f"{label} Poison Nodes - Average Eigenvector Centrality: {avg_eigenvector}")

    partition = detect_communities(homo_g)
    community_counts = {}
    for node in poison_nodes:
        community = partition.get(node, -1)
        community_counts[community] = community_counts.get(community, 0) + 1
    print(f"{label} Poison Nodes - Community Distribution: {community_counts}")

    plot_community_distribution(community_counts, label)

    clustering_coeffs, core_numbers = analyze_local_subgraphs(homo_g)
    avg_clustering = np.mean([clustering_coeffs.get(node, 0) for node in poison_nodes])
    avg_core_number = np.mean([core_numbers.get(node, 0) for node in poison_nodes])
    print(f"{label} Poison Nodes - Average Clustering Coefficient: {avg_clustering}")
    print(f"{label} Poison Nodes - Average Core Number: {avg_core_number}")


def calculate_centrality_measures(homo_g):
    centrality_measures = {}
    betweenness_file = "analyse_variable/betweenness.pkl"
    if os.path.exists(betweenness_file):
        with open(betweenness_file, "rb") as f:
            betweenness = pickle.load(f)
    else:
        print("Calculating betweenness centrality...")
        betweenness = nx.betweenness_centrality(homo_g)
        with open(betweenness_file, "wb") as f:
            pickle.dump(betweenness, f)

    closeness_file = "analyse_variable/closeness.pkl"
    if os.path.exists(closeness_file):
        with open(closeness_file, "rb") as f:
            closeness = pickle.load(f)
    else:
        print("Calculating closeness centrality...")
        closeness = nx.closeness_centrality(homo_g)
        with open(closeness_file, "wb") as f:
            pickle.dump(closeness, f)

    eigenvector_file = "analyse_variable/eigenvector.pkl"
    if os.path.exists(eigenvector_file):
        with open(eigenvector_file, "rb") as f:
            eigenvector = pickle.load(f)
    else:
        print("Calculating eigenvector centrality...")
        eigenvector = nx.eigenvector_centrality(homo_g, max_iter=1000)
        with open(eigenvector_file, "wb") as f:
            pickle.dump(eigenvector, f)

    centrality_measures['betweenness'] = betweenness
    centrality_measures['closeness'] = closeness
    centrality_measures['eigenvector'] = eigenvector

    return centrality_measures


def detect_communities(homo_g):
    partition_file = "analyse_variable/partition.pkl"
    if os.path.exists(partition_file):
        with open(partition_file, "rb") as f:
            partition = pickle.load(f)
    else:
        print("Detecting communities using Louvain algorithm...")
        from community import community_louvain
        partition = community_louvain.best_partition(homo_g)
        with open(partition_file, "wb") as f:
            pickle.dump(partition, f)
    return partition


def calculate_shortest_paths(homo_g, source_nodes, target_nodes):
    lengths_file = "analyse_variable/shortest_paths.pkl"
    if os.path.exists(lengths_file):
        with open(lengths_file, "rb") as f:
            lengths = pickle.load(f)
    else:
        print("Calculating shortest paths...")
        lengths = {}
        for source in source_nodes:
            lengths[source] = {}
            for target in target_nodes:
                try:
                    length = nx.shortest_path_length(homo_g, source=source, target=target)
                    lengths[source][target] = length
                except nx.NetworkXNoPath:
                    lengths[source][target] = float('inf')
        with open(lengths_file, "wb") as f:
            pickle.dump(lengths, f)
    return lengths


def analyze_local_subgraphs(homo_g):
    clustering_file = "analyse_variable/clustering_coeffs.pkl"
    if os.path.exists(clustering_file):
        with open(clustering_file, "rb") as f:
            clustering_coeffs = pickle.load(f)
    else:
        print("Calculating clustering coefficients...")
        clustering_coeffs = nx.clustering(homo_g)
        with open(clustering_file, "wb") as f:
            pickle.dump(clustering_coeffs, f)

    core_numbers_file = "analyse_variable/core_numbers.pkl"
    if os.path.exists(core_numbers_file):
        with open(core_numbers_file, "rb") as f:
            core_numbers = pickle.load(f)
    else:
        print("Calculating core numbers...")
        core_numbers = nx.core_number(homo_g)
        with open(core_numbers_file, "wb") as f:
            pickle.dump(core_numbers, f)

    return clustering_coeffs, core_numbers


def plot_community_distribution(community_counts, label):
    communities = list(community_counts.keys())
    counts = list(community_counts.values())

    plt.figure(figsize=(10, 6))
    sns.barplot(x=communities, y=counts, palette="viridis")
    plt.title(f"{label} Poison Nodes - Community Distribution")
    plt.xlabel("Community ID")
    plt.ylabel("Number of Nodes")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_multiple_community_distributions(community_counts_list, labels):

    all_communities = set()
    for counts in community_counts_list:
        all_communities.update(counts.keys())
    all_communities = sorted(list(all_communities))

    data = []
    for counts, label in zip(community_counts_list, labels):
        for community in all_communities:
            count = counts.get(community, 0)
            data.append({'Community': community, 'Count': count, 'Label': label})
    df = pd.DataFrame(data)

    plt.figure(figsize=(12, 6))
    sns.barplot(x='Community', y='Count', hue='Label', data=df)
    plt.title("Poison Nodes - Community Distribution Comparison")
    plt.xlabel("Community ID")
    plt.ylabel("Number of Nodes")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
