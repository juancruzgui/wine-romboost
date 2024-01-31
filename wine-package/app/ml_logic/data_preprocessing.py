import pandas as pandas
import numpy as np


def remove_outliers_iqr(df, columns=None):
    """
    Remove outliers from a DataFrame based on the Interquartile Range (IQR).

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - columns (list): List of columns to consider. If None, all numeric columns will be considered.

    Returns:
    - pd.DataFrame: DataFrame without outliers.
    """

    if columns is None:
        columns = df.select_dtypes(include='number').columns.tolist()

    filtered_df = df.copy()

    for column in columns:
        # Calculate quartiles
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)

        # Calculate IQR
        IQR = Q3 - Q1

        # Define upper and lower bounds to identify outliers
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Remove outliers
        filtered_df = filtered_df[(filtered_df[column] >= lower_bound) & (filtered_df[column] <= upper_bound)]

    return filtered_df

def drop_correlated_features(df_cleaned, threshold=0.8):
    """
    Clean Df by removing columns highly correlated. Calculates correlation
    matrix and features highly correlated (correlation>threshold), and then
    drops one of the features in high correlated features pairs.

    Parameters:
    - df_cleaned(pd.DataFrame): Input DataFrame
    - threshold(float<1): threshold for correlation

    Returns:
    - pd.DataFrame: DataFrame without highly correlated features

    """
    if threshold<1:
        df = df_cleaned.copy()
        correlation_matrix = df.corr()
        high_correlated = ((correlation_matrix > threshold) & (correlation_matrix < 1))

        # Use np.where to get the indices of True values in the boolean mask
        rows, columns = np.where(high_correlated)

        # Extract feature names from the indices and convert them to tuples
        feature_names = df_cleaned.columns
        correlated_features = [tuple(sorted([feature_names[rows[i]], feature_names[columns[i]]])) for i in range(len(rows))]
        #correlated pairs
        correlated_pairs = list(set(correlated_features))

        for pair in correlated_pairs:
            df.drop(columns=pair[0],inplace=True)
    else:
        print("Threshold has to be a value lower than 1")
        return None

    return df
