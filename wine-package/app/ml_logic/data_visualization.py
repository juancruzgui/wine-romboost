import matplotlib.pyplot as plt
import seaborn as sns
import math
import warnings
from matplotlib.colors import ListedColormap
import pandas as pd
import numpy as np
import matplotlib.patches as mpatches
import sys
import os
script_path = os.path.abspath(__file__)

cluster_colors = ['#ffc7ff', '#6c35de', '#4d425f', '#cd7e59', '#ddb247', '#d15252']
cluster_map = {key:value for key,value in enumerate(cluster_colors)}
cluster_mapping = ListedColormap(cluster_colors)
sns.set_palette(cluster_colors)
sns.set(rc={'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans']})



def plot_boxplots(df):
    """
    Plot boxplots for all wine DataFrame columns

    Parameters:
    - df(pd.DataFrame): Input DataFrame

    Returns:
    - None
    """
    nrows = math.ceil(len(df.columns)/3)
    ncols = min(len(df.columns),3)
    plt.figure(figsize=(ncols*4,nrows*2))

    plt.suptitle("\nWine Features Boxplot", fontsize=14)
    for i,col in enumerate(df.columns):
        plt.subplot(nrows,ncols,i+1);
        sns.boxplot(x=df[col],color='#6c35de')
        plt.xlabel(col, fontsize=10)
    plt.tight_layout()
    path = os.path.join(os.path.dirname(script_path),'..','images',f'wine_boxplots.png')
    plt.savefig(path)
    #plt.show()

def plot_distplots(df):
    """
    Plot distplots for all Wine DataFrame columns

    Parameters:
    - df(pd.DataFrame): Input DataFrame

    Returns:
    - None
    """
    nrow=max(math.ceil(len(df.columns)/3),1)
    ncols = min(len(df.columns),3)
    fig, axes = plt.subplots(nrows=nrow, ncols=3, figsize=(ncols*5,nrow*2.5))
    plt.suptitle("\nWine Features Distplot\n", fontsize=20)

    axes=axes.flatten()
    n_to_remove = 3-round(round((len(df.columns)/3)%1,2)*3)
    for i in range(1,n_to_remove+1):
        axes[-i].remove()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        for index, col in enumerate(df.columns):
            sns.distplot(x=df[col],ax=axes[index], bins=40,color='#4d425f')
            axes[index].set_xlabel(f'{col}')

    plt.tight_layout()
    path = os.path.join(os.path.dirname(script_path),'..','images',f'wine_distplots.png')
    plt.savefig(path)



def plot_correlations(df_cleaned):
    """
    Plot correlation matrix for Wine DataFrame

    Parameters:
    - df_cleaned(pd.DataFrame): Input DataFrame

    Returns:
    - Correlation Matrix
    """
    correlation_matrix = df_cleaned.corr()

    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, cmap='coolwarm', linewidths=.5, annot=True, annot_kws={'fontsize':8}, fmt = ".2f")

    # Customize the plot
    plt.title('\nCorrelation Heatmap', fontsize=20, pad=20)
    path = os.path.join(os.path.dirname(script_path),'..','images',f'wine_correlations.png')
    plt.savefig(path)

    return correlation_matrix

def plot_correlated_scatters(df,lower_bound, upper_bound):
    """
    Plot scatter plots for features correlated to each other, with
    a correlation between {lower_bound} and {upper_bound}

    Parameters:
    - df(pd.DataFrame): Input DataFrame
    - lower_bound(float)<1: lower bound for correlation between features
    - upper_bound(float)<1 and upper_bound(float)>lower_bound: upper bound for
    correlation between features

    Returns:
    - None
    """
    correlation_matrix = df.corr()
    filtered_correlation_matrix = correlation_matrix[((correlation_matrix>=lower_bound) & (correlation_matrix<upper_bound)) \
        | ((correlation_matrix<=-1*lower_bound) & (correlation_matrix>-1*upper_bound))]

    cleaned_matrix = filtered_correlation_matrix.dropna(how='all').dropna(axis=1, how='all')
    pairs = []
    i=0
    for index, row in cleaned_matrix.iterrows():
        for j,col in enumerate(cleaned_matrix.columns):
            if j<i:
                if (row[col] > 0) | (row[col] < 0) :
                    pairs.append((col,index, row[col]))
            else:
                pass
        i+=1
    nrow=max(math.ceil(len(pairs)/3),1)
    fig, axes = plt.subplots(nrows=nrow, ncols=3, figsize=(10,nrow*3.5))
    fig.suptitle(f'\nWine Features Scatter Plots ({lower_bound}<=|corr|<{upper_bound})\n', fontsize=16)

    axes=axes.flatten()
    if round(round((len(pairs)/3)%1,2)*3) == 0:
        n_to_remove = 0
    else:
        n_to_remove = 3 - int(round(round((len(pairs)/3)%1,2)*3))
    for i in range(1,n_to_remove+1):
        axes[-i].remove()

    for index, pair in enumerate(pairs):
        sns.scatterplot(data=df, x=pair[0],y=pair[1],ax=axes[index],s=50, edgecolor=None, alpha=0.7,zorder=2)

        axes[index].set_title(f'{pair[0]}\nvs\n{pair[1]}\ncorr = {round(pair[2],2)}', fontsize=12)
        axes[index].set_facecolor('#f0f0f0')
        axes[index].set_xlabel(pair[0], fontsize=10)
        axes[index].set_ylabel(pair[1],fontsize=10)
        # Add grid lines
        axes[index].grid(True, linestyle='-', alpha=0.7, color='white', zorder=1, linewidth=1.5)
        # Remove border lines
        for spine in axes[index].spines.values():
            spine.set_visible(False)

    plt.tight_layout()
    path = os.path.join(os.path.dirname(script_path),'..','images',f'wine_correlated_scatters({lower_bound},{upper_bound}).png')
    plt.savefig(path)
    return cleaned_matrix

def plot_clusters(X:pd.DataFrame,km,title=None,random_state=42,subplot=False,nrows=1,ncols=1,index=1):
    """
    Plots scatterplots for first two variables on X distinguishing
    between different clusters

    Parameters:
    - X(pd.DataFrame): Input dataframe
    - km(Kmeans() object): Kmeans initialized object
    - title(str): Title for the plots
    - subplot
    - nrows
    - ncols
    - index

    Returns:

    """
    cluster_assignments = km.labels_
    k = len(list(set(cluster_assignments)))
    if subplot==False:
        plt.figure(figsize=(6,4))
    else:
        plt.subplot(nrows,ncols,index);


    plt.suptitle(title, fontsize=14)
    plt.title(f"KMeans Clustering (k={k})", fontsize=12)
    sns.scatterplot(data=X, x=X.columns[0],y=X.columns[1],hue=cluster_assignments,s=25, edgecolor=None, palette=cluster_map)
    plt.tight_layout()
    path = os.path.join(os.path.dirname(script_path),'..','images',f'wine_clusters_plot-k{k}.png')
    plt.savefig(path)


def biplot(score:pd.DataFrame, cluster_labels, coeff, labels=None):
    """
    Plots biplot to check variable influences on clusters

    Parameters:
    - score(pd.DataFrame): PCA dataframe
    - cluster_labels(np.array()): array with cluster labels
    - coeff: PCA coefficients

    """
    plt.figure(figsize=(8, 8))
    plt.title("\nPCA Biplot\n")
    xs = score.loc[:, score.columns[0]]
    ys = score.loc[:, score.columns[1]]
    n = coeff.shape[0]

    scalex = 1.0 / (xs.max() - xs.min())
    scaley = 1.0 / (ys.max() - ys.min())

    sns.scatterplot(x= xs * scalex, y = ys * scaley, hue = cluster_labels, palette=cluster_map)

    for i in range(n):
        plt.arrow(0, 0, coeff[i, 0], coeff[i, 1], color='r', alpha=0.5)
        if labels is None:
            plt.text(coeff[i, 0] * 1.15, coeff[i, 1] * 1.15, "Var" + str(i + 1), color='g', ha='center', va='center')
        else:
            plt.text(coeff[i, 0] * 1.15, coeff[i, 1] * 1.15, labels[i], color='g', ha='center', va='center')

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid()
    path = os.path.join(os.path.dirname(script_path),'..','images',f'PCA_biplot.png')
    plt.savefig(path)


def pairplot(df:pd.DataFrame,title=None):
    """
    Plot scatterplots between all variables and histplots for all variables
    distinguishing between clusters

    Parameters:
    - df(pd.DataFrame): wine dataframe
    - title: plot suptitle
    """
    sns.pairplot(df, hue='cluster', palette=cluster_map)
    plt.suptitle(title, fontsize=18)
    plt.tight_layout()
    path = os.path.join(os.path.dirname(script_path),'..','images',f'wine_pairplot.png')
    plt.savefig(path)


def cluster_comparison_bar(X_comparison:pd.DataFrame, colors, title="Cluster results"):
    """
    Plot comparisson of the mean per cluster to overall mean in percent for
    each variable

    Parameters:
    - X_comparison(pd.DataFrame): dataframe where each index is a feature and
    columns represent the devation from the overall mean for each cluster
    - colors(ls): list of colors for bars
    - title: suptitle for the plot

    """
    features = X_comparison.index
    ncols = 3
    clusters = len(X_comparison.columns)
    # calculate number of rows
    nrows = len(features) // ncols + (len(features) % ncols > 0)
    # set figure size
    fig = plt.figure(figsize=(15,15), dpi=200)
    #interate through every feature
    for n, feature in enumerate(features):
        # create chart
        ax = plt.subplot(nrows, ncols, n + 1)
        X_comparison[X_comparison.index==feature].plot(kind='bar', ax=ax, title=feature,
                                                             color=colors[0:clusters],
                                                             legend=False
                                                            )
        plt.axhline(y=0)
        x_axis = ax.axes.get_xaxis()
        x_axis.set_visible(False)

    c_labels = X_comparison.columns.to_list()
    c_colors = colors[0:3]
    mpats = [mpatches.Patch(color=c, label=l) for c,l in list(zip(colors[0:clusters],
                                                                  X_comparison.columns.to_list()))]

    fig.legend(handles=mpats,
               ncol=ncols,
               loc="upper center",
               fancybox=True,
               bbox_to_anchor=(0.5, 0.96)
              )
    axes = fig.get_axes()

    fig.suptitle(title, fontsize=18, y=1)
    fig.supylabel('Deviation from overall mean in %')
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    path = os.path.join(os.path.dirname(script_path),'..','images',f'wine_cluster_mean_deviations.png')
    plt.savefig(path)


def cluster_characteristics(X_dev_rel, title="\nCluster characteristics\n"):
    """
    Plot wine deviation from overall mean for each wine feature

    Parameters:
    - X_dev_rel(pd.DataFrame): Dataframe of features and deviation from overall
    mean for each cluster
    - title(str)
    """

    colors = ['#9EBD6E','#81a094','#775b59','#32161f', '#946846', '#E3C16F', '#fe938c', '#E6B89C','#EAD2AC',
          '#DE9E36', '#4281A4','#37323E','#95818D'
         ]

    fig = plt.figure(figsize=(10,5), dpi=200)
    X_dev_rel.T.plot(kind='bar',
                        ax=fig.add_subplot(),
                        title="\nCluster characteristics\n",
                        color=colors,
                        xlabel="Cluster",
                        ylabel="Deviation from overall mean in %"
                        )
    plt.axhline(y=0, linewidth=1, ls='--', color='black')
    plt.legend(bbox_to_anchor=(1.04,1))
    fig.autofmt_xdate(rotation=0)
    plt.tight_layout()
    path = os.path.join(os.path.dirname(script_path),'..','images',f'wine_cluster_characteristics.png')
    plt.savefig(path)


class Radar(object):
    """
    Radar object
    """
    def __init__(self, figure,title, labels, rect=None, ):
        if rect is None:
            rect = [0.05, 0.05, 0.9, 0.9]

        self.n = len(title)
        self.angles = np.arange(0, 360, 360.0/self.n)

        self.axes = [figure.add_axes(rect, projection='polar', label='axes%d' % i) for i in range(self.n)]
        self.ax = self.axes[0]
        self.ax.set_thetagrids(self.angles, labels=title, fontsize=14, backgroundcolor="white",zorder=999) # Feature names
        self.ax.set_yticklabels([])

        for ax in self.axes[1:]:
            ax.xaxis.set_visible(False)
            ax.set_yticklabels([])
            ax.set_zorder(-99)

        for ax, angle, label in zip(self.axes, self.angles, labels):
            ax.spines['polar'].set_color('black')
            ax.spines['polar'].set_zorder(-99)

    def plot(self, values, *args, **kw):
        angle = np.deg2rad(np.r_[self.angles, self.angles[0]])
        values = np.r_[values, values[0]]
        self.ax.plot(angle, values, *args, **kw)
        kw['label'] = '_noLabel'
        self.ax.fill(angle, values,*args,**kw)

def radar_plot(km, X_std_mean):
    """
    Plot radar chart for wine attributes and clusters.

    Parameters:
    - km(Kmeans()): Kmeans object
    - X_std_mean(pd.DataFrame)
    """
    fig = plt.figure(figsize=(8, 10))
    no_features = len(km.feature_names_in_)
    radar = Radar(fig, km.feature_names_in_, labels = np.unique(km.feature_names_in_))

    for k in range(0,km.n_clusters):
        cluster_data = X_std_mean[k].values.tolist()
        radar.plot(cluster_data,  '-', lw=2, color=cluster_colors[k], alpha=0.7, label='cluster {}'.format(k))

    radar.ax.legend(fancybox=True,
                bbox_to_anchor=(1.04, 0.99)
                )
    radar.ax.set_title("\nCluster characteristics:\n", size=18, pad=60)
    fig.subplots_adjust(left=0.2, right=0.8, top=0.9, bottom=0.1)
    path = os.path.join(os.path.dirname(script_path),'..','images',f'wine_cluster_radar.png')
    plt.savefig(path)
