import matplotlib.pyplot as plt
import seaborn as sns
import math
import pandas as pd

def plot_boxplots(df):
    """
    Plot boxplots for all wine DataFrame columns

    Parameters:
    - df(pd.DataFrame): Input DataFrame

    Returns:
    - None
    """
    sns.set(rc={'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans']})
    nrows = math.ceil(len(df.columns)/3)
    ncols = min(len(df.columns),3)
    plt.figure(figsize=(ncols*4,nrows*2))

    plt.suptitle("\nWine Features Boxplot", fontsize=14)
    for i,col in enumerate(df.columns):
        plt.subplot(nrows,ncols,i+1);
        sns.boxplot(x=df[col],color='purple')
        plt.xlabel(col, fontsize=10)

    return None

def plot_distplots(df):
    """
    Plot distplots for all Wine DataFrame columns

    Parameters:
    - df(pd.DataFrame): Input DataFrame

    Returns:
    - None
    """
    sns.set(rc={'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans']})
    nrow=max(math.ceil(len(df.columns)/3),1)
    ncols = min(len(df.columns),3)
    fig, axes = plt.subplots(nrows=nrow, ncols=3, figsize=(ncols*5,nrow*2.5))
    plt.suptitle("\nWine Features Distplot\n", fontsize=20)

    axes=axes.flatten()
    n_to_remove = 3-round(round((len(df.columns)/3)%1,2)*3)
    for i in range(1,n_to_remove+1):
        axes[-i].remove()

    for index, col in enumerate(df.columns):
        sns.distplot(x=df[col],ax=axes[index], bins=40,color='blue')
        axes[index].set_xlabel(f'{col}')

    plt.tight_layout()
    plt.savefig('images/wine_distplots.png')
    plt.show();

    return None


def plot_correlations(df_cleaned):
    """
    Plot correlation matrix for Wine DataFrame

    Parameters:
    - df_cleaned(pd.DataFrame): Input DataFrame

    Returns:
    - Correlation Matrix
    """
    correlation_matrix = df_cleaned.corr()
    sns.set(rc={'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans']})

    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, cmap='coolwarm', linewidths=.5)

    # Customize the plot
    plt.title('\nCorrelation Heatmap', fontsize=20, pad=20)
    plt.savefig('images/wine_correlations.png')
    plt.show()


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

    sns.set(rc={'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans']})
    correlation_matrix = df.corr()
    filtered_correlation_matrix = correlation_matrix[(correlation_matrix>lower_bound) & (correlation_matrix<upper_bound)]

    cleaned_matrix = filtered_correlation_matrix.dropna(how='all').dropna(axis=1, how='all')
    pairs = []
    i=0
    for index, row in cleaned_matrix.iterrows():
        for j,col in enumerate(cleaned_matrix.columns):
            if j>i:
                if row[col] > 0:
                    pairs.append((index,col))
            else:
                pass
        i+=1
    nrow=max(math.ceil(len(pairs)/3),1)
    fig, axes = plt.subplots(nrows=nrow, ncols=3, figsize=(10,nrow*3))
    fig.suptitle(f'\nWine Features Scatter Plots ({lower_bound}<corr<{upper_bound})\n', fontsize=16)

    axes=axes.flatten()
    n_to_remove = 3-round(round((len(pairs)/3)%1,2)*3)
    for i in range(1,n_to_remove+1):
        axes[-i].remove()

    for index, pair in enumerate(pairs):
        sns.scatterplot(data=df, x=pair[0],y=pair[1],ax=axes[index], color='purple',s=50, edgecolor=None, alpha=0.7,zorder=2)

        axes[index].set_title(f'{pair[0]} vs {pair[1]}', fontsize=12)
        axes[index].set_facecolor('#f0f0f0')
        axes[index].set_xlabel(pair[0], fontsize=10)
        axes[index].set_ylabel(pair[1],fontsize=10)
        # Add grid lines
        axes[index].grid(True, linestyle='-', alpha=0.7, color='white', zorder=1, linewidth=1.5)
        # Remove border lines
        for spine in axes[index].spines.values():
            spine.set_visible(False)

    plt.tight_layout()
    plt.savefig(f'images/wine_correlated_scatters({lower_bound},{upper_bound}).png')
    plt.show();
