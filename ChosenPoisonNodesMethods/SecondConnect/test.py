import numpy as np
import random


def reduce_matrix_size(data_matrix, top_k_cols=100, sample_rows=100, seed=0):

    np.random.seed(seed)
    random.seed(seed)

    num_rows, num_cols = data_matrix.shape

    col_sums = data_matrix.sum(axis=0) 
    col_indices = np.argsort(col_sums)[::-1][:top_k_cols]

    reduced_matrix_cols = data_matrix[:, col_indices]

    if sample_rows < num_rows:
        row_indices = random.sample(range(num_rows), sample_rows)
        reduced_matrix = reduced_matrix_cols[row_indices, :]
    else:
        row_indices = list(range(num_rows))
        reduced_matrix = reduced_matrix_cols

    return reduced_matrix, row_indices, col_indices


if __name__ == "__main__":
    data_matrix = np.random.rand(10000, 20000)

    small_mat, used_rows, used_cols = reduce_matrix_size(data_matrix, top_k_cols=100, sample_rows=100)

    import seaborn as sns
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 10))
    sns.heatmap(small_mat, cmap="YlGnBu")
    plt.show()
