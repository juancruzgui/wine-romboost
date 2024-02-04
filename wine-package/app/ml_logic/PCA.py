from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

def plot_pca_n_decision(X_scaled):
    """
    Plots explained variance vs PCA number of components

    Parameters:
    - X_scaled(np.Array): Scaled input variables
    """

    #selection the correct number of components
    pca = PCA()
    pca.fit(X_scaled)

    # Explained variance for each component
    explained_variance = pca.explained_variance_ratio_

    # Cumulative explained variance
    cumulative_variance = np.cumsum(explained_variance)

    # Plot explained variance
    plt.plot(range(1, len(explained_variance) + 1), cumulative_variance, marker='o', linestyle='-', color='b')
    plt.xlabel('Number of Components', fontsize=10)
    plt.ylabel('Cumulative Explained Variance', fontsize=10)
    plt.title('\nExplained Variance vs. Number of Components\n', fontsize=14)
    plt.savefig(f'images/explained_variance_PCA.png')

    return cumulative_variance
