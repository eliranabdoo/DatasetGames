from functools import wraps

import matplotlib.pyplot as plt
import numpy as np

NUMERIC_TYPES = ['float', 'float64', 'int', 'int64']


def plotable(func):
    @wraps(func)
    def wrapper(*args, show=True, **kwargs):
        res = func(*args, **kwargs)
        if show:
            plt.show()
        return res

    return wrapper


@plotable
def calc_correlations_with_target(df, target_col, feature_cols=None):
    """Calculate correlations of all numeric columns with target col, assumes target col is numerical
    We refer to the absolute value of pearson correlation, since negative correlation is just as predictive"""
    res = {}
    if feature_cols is not None:
        features = feature_cols
    else:
        numeric_cols = [col for col in df.columns if col != target_col and df[col].dtype in NUMERIC_TYPES]
        features = numeric_cols
    for col in features:
        res[col] = np.abs(df[target_col].corr(df[col]))

    plt.bar(x=list(res.keys()), height=list(res.values()))
    plt.xticks(rotation=45, ha="right")
    plt.title("Correlations to target")
    return res


@plotable
def target_means_per_feature(df, target_col, feature_cols, sort=True):
    """Plot target means per value, for each of the given features"""
    n = len(feature_cols)
    res = {}
    plt.figure(1)
    for i in range(n):
        plt.subplot(n, 1, i + 1)
        curr_col = feature_cols[i]
        mean = df.groupby(df[curr_col])[target_col].mean()
        if sort:
            mean = mean.sort_values()
        res[curr_col] = mean
        mean.plot(subplots=True, kind='bar', use_index=True)
    plt.subplots_adjust(hspace=0.8)
    return res


@plotable
def target_std_per_feature(df, target_col, feature_cols):
    """Map features to target means standard deviations
    Can be used to check how imbalanced is the target with respect to a feature"""
    n = len(feature_cols)
    res = {}
    plt.figure(1)
    for i in range(n):
        curr_col = feature_cols[i]
        std = df.groupby(df[curr_col])[target_col].mean().std()
        res[curr_col] = std
    plt.title('Standard deviations for features')
    plt.bar(range(len(res)), res.values(), align='center')
    plt.xticks(range(len(res)), list(res.keys()))
    return res
