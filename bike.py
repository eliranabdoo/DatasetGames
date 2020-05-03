import sklearn
import pandas as pd
import os
import seaborn as sns
from sklearn.base import TransformerMixin
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from numpy.core.defchararray import zfill
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import f_regression, SelectKBest, mutual_info_classif, mutual_info_regression
from sklearn.pipeline import Pipeline

data_folder = "Bike-Sharing-Dataset"

data_files = {'day': 'day.csv', 'hour': 'hour.csv'}
date_col = 'dteday'
hour_col = 'hr'
full_date_col = 'fulldt'
target_col = 'cnt'


def calc_target_corr(df, target_col):
    """Calculate correlations of all numeric columns with target col, assumes target col is numerical"""
    res = {}
    for col in df.columns:
        if col != target_col and df[col].dtype in ['float', 'float64', 'int', 'int64']:
            res[col] = np.abs(df[target_col].corr(df[col]))
    return res


def target_stats_by_cols(df, target_col, feature_cols):
    """Print averages of target per feature values"""
    n = len(feature_cols)
    plt.figure(1)
    res = {}
    for i in range(n):
        plt.subplot(n, 1, i + 1)
        curr_col = feature_cols[i]
        mean = df.groupby(df[curr_col])[target_col].mean().sort_values()
        mean.plot(subplots=True, kind='bar', title='Avg By %s' % curr_col, use_index=True)
        std = mean.std()
        res[curr_col] = {'std': std}
    plt.show()
    return res


class FeatureDropper(TransformerMixin):

    def __init__(self, cols):
        self.cols = cols

    def fit(self, X=None, y=None):
        return self

    def transform(self, X, y=None):
        return X.drop(columns=self.cols)


class FeatureSelector(TransformerMixin):

    def __init__(self, cols):
        self.cols = cols

    def fit(self, X=None, y=None):
        return self

    def transform(self, X, y=None):
        return X[self.cols]


class RemoveDateTimes(TransformerMixin):

    def fit(self, X=None, y=None):
        return self

    def transform(self, X, y=None):
        return X.drop(columns=list(X.select_dtypes(include=['datetime64'])))


print(os.getcwd())
for name, file in data_files.items():
    path = os.path.join(data_folder, file)
    locals()[name + '_orig'] = pd.read_csv(path, index_col=0)

for name in data_files:
    locals()[name] = locals()[name + '_orig'].copy()

locals_ = locals()
datasets = [locals_[name] for name in data_files]

#  EDA
for ds in datasets:
    ds[date_col] = pd.to_datetime(ds[date_col])

# Examine differences between day and hour data
hour.insert(loc=len(hour.columns), column=full_date_col,
            value=pd.to_datetime(
                hour[date_col].astype(str) + '-' + hour[hour_col].astype(str).apply(zfill,
                                                                                    width=2),
                format='%Y-%m-%d-%H'))

print(day[target_col].sum())
print(hour.resample('D', on=full_date_col)[target_col].sum().sum())
# both show identical target values, so hourly data is a decomposed version of the daily data

# Find pearson coefficients between features and target
corrs = calc_target_corr(day, target_col)
print(corrs)
# plt.bar(x=list(corrs.keys()), height=list(corrs.values()))
# plt.show()

# Registered and casual are irrelevant as they form target leakage. Will not be available in prediction time
# temp and season are redundant due to month and atemp, which was found as slightly better correlating to target
leaking_columns = ['registered', 'casual']
redundant_columns = ['season', 'temp']
irrelevant_columns = leaking_columns + redundant_columns
low_correlation_threshold = 0.1  # features with lower correlation are discarded
relevant_corrs = dict(
    filter(lambda x: x[0] not in irrelevant_columns and x[1] > low_correlation_threshold, corrs.items()))
relevant_features = list(relevant_corrs.keys())

sns.pairplot(day[relevant_features + [target_col]], kind='scatter')
# plt.show()

print(target_stats_by_cols(day, target_col, relevant_features))

# splitting
day_shuffled = day.sample(frac=1)
X = day_shuffled.drop(columns=[target_col])
y = day_shuffled[target_col]

train_size = 0.85
valid_size = 0.2  # fold size in cross-validation

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 - train_size, random_state=37)

# Different feature selections, comparing my manually selected to auto selected using two methods
my_fs = Pipeline(steps=[
    ('select_features', FeatureSelector(relevant_features))
])
k1, k2 = 10, 10
freg_fs = Pipeline(steps=[
    ('select_features', SelectKBest(score_func=f_regression, k=k1))
])

mi_fs = Pipeline(steps=[
    ('select_features', SelectKBest(score_func=mutual_info_regression, k=k2))
])

feature_selectors = [my_fs, freg_fs, mi_fs]

best_fs_pipeline = None
best_fs_score = -np.inf
for fs in feature_selectors:
    curr_pipeline = Pipeline(steps=[
        ('remove_leakage', FeatureDropper(leaking_columns)),
        ('remove_datetime', RemoveDateTimes()),
        ('select_features', fs),
        ('model', RandomForestRegressor(n_estimators=10))
    ])
    curr_score = cross_val_score(X=X_train, y=y_train, estimator=curr_pipeline, cv=int(1 / valid_size)).mean()

    # try:  Problematic, the returned indices are for the preprocessed data
    #     print("Chose %s" % str(X_train.columns[curr_pipeline.steps[2][1].steps[0][1].get_support(indices=True)]))
    # except Exception as e:
    #     print("Chose %s" % str(curr_pipeline.steps[2][1].steps[0][1].cols))
    print("Average cv score: %f" % curr_score)
    if curr_score > best_fs_score:
        best_fs_score = curr_score
        best_fs_pipeline = curr_pipeline

baseline_model = best_fs_pipeline
