from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
script_path = os.path.abspath(__file__)

def k_test(X, k_tests:list):
    """
    Calculates Within-Cluster-Sum-of-Squares and Silhouette score for a list of
    different k numbers of clusters for KMeans algorithm, and plots both graphs
    to assist visually on the decision of the number of clusters to use.

    Input:
    - X(pd.DataFrame or np.array): Input features DataFrame
    - k_tests(list): List of k number of clusters to try

    Returns:
    - dict: Dictionary with the wcss and silo scores for different k number of
    clusters

    Example:
        k_test(df,[2,3,4,5,6])
        output:
            {'k': {2: {'wcss': 1420.516853221093, 'silo': 0.24694353961552332},
                    3: {'wcss': 1044.667516683622, 'silo': 0.2934200205200105},
                    4: {'wcss': 975.6666320656269, 'silo': 0.2468556636432967},
                    5: {'wcss': 910.6438527286055, 'silo': 0.2034142734655438},
                    6: {'wcss': 855.1459149651747, 'silo': 0.14551805468551002}}}
    """

    wcss = []; silo = []
    random_state=42
    for k in k_tests:
        clustering = KMeans(k,random_state=random_state).fit(X)
        wcss.append(clustering.inertia_)
        silo.append(silhouette_score(X,clustering.predict(X)))

    tests_dict = {'k':{
        key:{'wcss':wcss[i],
             'silo':silo[i]} for i,key in enumerate(k_tests)}
    }


    plt.figure(figsize=(10,4))
    plt.suptitle("\nNumber of cluster decision throught Elbow Method and Siloutte Score", fontsize=14)

    plt.subplot(1,2,1);
    plt.title("Within-Cluster-Sum-of-Squares", fontsize=12)
    plt.scatter(k_tests, wcss, marker='o', edgecolor='k')
    plt.plot(k_tests, wcss, linestyle='--')
    plt.xlabel('Number of Clusters (k)', fontsize=10)
    plt.ylabel('Within-Cluster Sum of Squares (WCSS)', fontsize=10)

    plt.subplot(1,2,2);
    plt.title("Silhouette Width", fontsize=12)
    plt.scatter(k_tests, silo, marker='o', edgecolor='k')
    plt.plot(k_tests, silo, linestyle='--')
    plt.xlabel('Number of Clusters (k)', fontsize=10)
    plt.ylabel('Silhouette Score)', fontsize=10)

    plt.tight_layout()
    path = os.path.join(os.path.dirname(script_path),'..','images',f'wine_cluster_tests-k.png')
    plt.savefig(path)

    return tests_dict

def k_means(X:pd.DataFrame,k=3,random_state=42):
    """
    Performs KMeans clustering algorithm with k number of clusters
    and returns the KMeans object

    Input:
    - X(pd.DataFrame): Input variables dataframe
    - k(int): number of desired_clusters
    - random_state(int)

    Returns
    - km(Kmeans())
    - km_fit(Kmeans().fit())
    - cluster_assingments(np.Array): cluster labels
    """
    km = KMeans(n_clusters=k,
            max_iter=300,
            tol=1e-04,
            init='k-means++',
            n_init=10,
            random_state=random_state)
    km_fit = km.fit(X)
    cluster_assignments = km.predict(X)
    return km,km_fit,cluster_assignments
