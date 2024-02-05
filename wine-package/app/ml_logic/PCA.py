from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
script_path = os.path.abspath(__file__)

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
    plt.figure(figsize=(8, 8))
    # Plot explained variance
    plt.plot(range(1, len(explained_variance) + 1), cumulative_variance, marker='o', linestyle='-', color='b')
    plt.xlabel('Number of Components', fontsize=10)
    plt.ylabel('Cumulative Explained Variance', fontsize=10)
    plt.title('\n\nExplained Variance vs. Number of Components\n', fontsize=14)
    path = os.path.join(os.path.dirname(script_path),'..','images',f'explained_variance_PCA.png')
    plt.savefig(path)

    return cumulative_variance
