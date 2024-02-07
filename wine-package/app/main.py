import requests
from io import BytesIO
import pandas as pd
from sklearn.preprocessing import StandardScaler
import ml_logic.clustering as clus
import ml_logic.data_preprocessing as prep
import ml_logic.data_visualization as viz
from ml_logic.data_visualization import cluster_colors, Radar
import ml_logic.PCA as pca
from sklearn.decomposition import PCA
import numpy as np


#API REQUEST
api_url = 'http://0.0.0.0:8000/wine-raw'

line = "-"*110

print("🪢Making API request")
# Make the API request
response = requests.get(api_url)

print(line)
# Check if the request was successful (status code 200)
if response.status_code == 200:
    # Read the file data into a BytesIO object
    file_data = BytesIO(response.content)

    # Use Pandas or any other library to process the file content as needed
    df = pd.read_csv(file_data)

    # Now 'df' contains the file content as a DataFrame, and you can use it as needed
    print(f"\n✅ Wine DataFrame retrieved successfully from {'http://localhost:8000/wine-raw'}")
    print(f"\n📐 Wine DataFrame shape {df.shape}\n")
else:
    # Print an error message if the request was not successful
    print(f"Error: {response.status_code} - {response.text}")


#DF
print("🍇 Dataframe:\n")
print(df.head())
print("\n")
print("🍷 Wine Attributes:\n")
print(df.info())
print("\n")

print(line)

#Outliers
df_cleaned = prep.remove_outliers_iqr(df)
print("\n👽 Outliers removed successfully")
viz.plot_boxplots(df_cleaned)
print("\n📦 Boxplots done\n")

print(line)



#Statistics
print("\n🧪 Statistics\n")
print(df_cleaned.describe())
viz.plot_distplots(df_cleaned)
print("\n")
print(line)



#Correlations
correlation_matrix = viz.plot_correlations(df_cleaned)

print(f"\n🖊️ Printing correlations\n")

print(f"\n- Correlations [0.4,0.6)")
medium_correlation = viz.plot_correlated_scatters(df_cleaned,0.4,0.6)
print(medium_correlation)

print(f"\n- Correlations [0.6,0.8)")
strong_correlation = viz.plot_correlated_scatters(df_cleaned,0.6,0.8)
print(strong_correlation)

print(f"\n- Correlations [0.8,1)")
very_strong_correlation = viz.plot_correlated_scatters(df_cleaned,0.8,1)
print(very_strong_correlation)

print("\n")
print(line)




#Clustering
print("\n📉 Clustering without PCA")
X = df_cleaned.copy()
X = pd.DataFrame(StandardScaler().fit_transform(X), columns = X.columns)

print(f'\n❎ Finding best number of clusters:')
ktests = clus.k_test(X, [2,3,4,5,6])
scores = []
ks = []
for key,value in ktests['k'].items():
    print(f'- {key} clusters WCSS={round(value["wcss"],2)} Silhouette Score={round(value["silo"],2)}')
    scores.append(value['silo'])
    ks.append(key)

k = ks[scores.index(max(scores))]
print(f'\n⚡ Number of clusters elected: {k}\n - Silhouette Score: {round(max(scores),2)}')

# -----Perform Kmeans clustering with k=3
km, km_fit, cluster_assigns = clus.k_means(X,k=3)
viz.plot_clusters(X,km, subplot=False)
print(line)




#PCA
print("\n📉 Clustering with PCA")
X=df_cleaned.copy()
X_transformed=StandardScaler().fit_transform(X)

cum_variance = pca.plot_pca_n_decision(X_transformed)
print("\nCumulative explained variance:")
for i,var in enumerate(cum_variance):
    print(f'- {i+1} PCA components - explained variance = {round(var,2)}')

pca = PCA(n_components=3).fit(X_transformed)
X_pca = pd.DataFrame(pca.transform(X_transformed), columns = [f'PC{i}' for i in range(3)])

print(f'\n❎ Finding best number of clusters:')
ktests = clus.k_test(X_pca, [2,3,4,5,6])
scores = []
ks = []
for key,value in ktests['k'].items():
    print(f'- {key} clusters WCSS={round(value["wcss"],2)} Silhouette Score={round(value["silo"],2)}')
    scores.append(value['silo'])
    ks.append(key)

k = ks[scores.index(max(scores))]
print(f'\n⚡ Number of clusters elected: {k}\n - Silhouette Score: {round(max(scores),2)}')

# -----Perform Kmeans clustering with k=3
km_pca,km_fit_pca, cluster_assignments = clus.k_means(X_pca)
viz.plot_clusters(X_pca,km_pca, title='PCA (n=3)')

print('\n📈 Printing clusters\n')
print('\n🪶 Printing biplot\n')
viz.biplot(X_pca, cluster_assignments, np.transpose(pca.components_), labels=df_cleaned.columns)
print('\n')
print(line)



#VARIABLE ANALYSIS
print('\n🪄 Analyzing Variables \n')
df_cleaned['cluster'] = cluster_assignments
print('\n🪶 Printing pairplot\n')
viz.pairplot(df_cleaned, '\nWine Features pairplot\n\n')

print('\n🔎 Analyzing mean deviation from overall mean for each feature:\n')
#Mean analysis
X['cluster'] = cluster_assignments
X_std = pd.DataFrame(X_transformed, columns = X.drop(columns=['cluster']).columns)
X_std['cluster'] = cluster_assignments

X_mean = pd.concat([pd.DataFrame(X.mean().drop('cluster'), columns=['mean']),
                   X.groupby('cluster').mean().T], axis=1)

X_dev_rel = X_mean.apply(lambda x: round((x-x['mean'])/x['mean'],2)*100, axis = 1)
X_dev_rel.drop(columns=['mean'], inplace=True)
X_mean.drop(columns=['mean'], inplace=True)

X_std_mean = pd.concat([pd.DataFrame(X_std.mean().drop('cluster'), columns=['mean']),
                   X_std.groupby('cluster').mean().T], axis=1)

X_std_dev_rel = X_std_mean.apply(lambda x: round((x-x['mean'])/x['mean'],2)*100, axis = 1)
X_std_dev_rel.drop(columns=['mean'], inplace=True)
X_std_mean.drop(columns=['mean'], inplace=True)
print(X_dev_rel)
print('\n')

print('\n🪶 Printing mean comparisson for features')
viz.cluster_comparison_bar(X_dev_rel, cluster_colors, title="\nComparison of the mean per cluster to overall mean in percent\n")

print('\n🪶 Printing cluster characteristics')
viz.cluster_characteristics(X_dev_rel, title="\nCluster characteristics\n")

print('\n🪶 Printing radar plot')
viz.radar_plot(km,X_std_mean)

print('\n✅ Wine Analysis done - Check images http://localhost:8000/analysis-images')
