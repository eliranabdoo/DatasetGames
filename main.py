from collections import defaultdict

import sklearn
import pandas as pd
import os
import seaborn as sns
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from numpy.core.defchararray import zfill
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import f_regression, SelectKBest, mutual_info_classif, mutual_info_regression
from sklearn.pipeline import Pipeline
import re

data_path = "./Imports/imports-85.data"
names_path = "./Imports/imports-85.names"

target_col = 'price'
attribute_match_regex = "\d{1,2}\. (.+?):"
missing_val = '?'


def get_features_from_names(path):
    with open(path) as f:
        attributes_section = False
        features = []
        for line in f.readlines():
            if "Attribute Information" in line:
                attributes_section = True
                continue

            if line == "\n":
                attributes_section = False
                continue

            if attributes_section:
                match = re.search(attribute_match_regex, line)
                if match:
                    features.append(match.group(1))
        return features


def get_numeric_from_names(path):
    with open(path) as f:
        attributes_section = False
        numeric_features = []
        for line in f.readlines():
            if "Attribute Information" in line:
                attributes_section = True
                continue

            if line == "\n":
                attributes_section = False
                continue

            if attributes_section:
                match = re.search(attribute_match_regex, line)
                if match and 'continuous' in line:
                    numeric_features.append(match.group(1))
        return numeric_features


def calc_target_corr(df, target_col):
    res = {}
    for col in df.columns:
        if col != target_col and df[col].dtype in ['float', 'float64', 'int', 'int64']:
            res[col] = np.abs(df[target_col].corr(df[col]))
    return res


def target_stats_by_cols(df, target_col, feature_cols):
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


class AverageImputer(TransformerMixin, BaseEstimator):
    def __init__(self, predictors_map=None):
        self.predictors_map = predictors_map
        self.source_allowed_types = ['object']
        self.target_allowed_types = ['int', 'int64', 'float', 'float64']
        self.imputer_maps = {}
        self.default_val_func = max

    def fit(self, X, y=None, predictors_map=None):
        self.predictors_map = predictors_map if predictors_map is not None else self.predictors_map
        target_predictor_pairs = list(self.predictors_map.items())
        source_cols = list(map(lambda x: x[1], target_predictor_pairs))
        target_cols = list(map(lambda x: x[0], target_predictor_pairs))
        source_allowed_cols = set(X.select_dtypes(include=self.source_allowed_types).columns)
        target_allowed_cols = set(X.select_dtypes(include=self.target_allowed_types).columns)

        assert set(source_cols).issubset(source_allowed_cols)
        assert set(target_cols).issubset(target_allowed_cols)

        for t_col in target_cols:
            impute_map = {}

            s_col = self.predictors_map[t_col]
            values = X[s_col].unique()
            for v in values:
                impute_map[v] = X.loc[X[t_col].notnull() & (X[s_col] == v), t_col].mean()
            self.imputer_maps[t_col] = impute_map

        for impute_map in self.imputer_maps.values():
            default_val = self.default_val_func([val for val in impute_map.values() if val is not np.nan])
            for v, avg in impute_map.items():
                if avg is np.nan:
                    impute_map[v] = default_val
        return self

    def transform(self, X):
        res = X.copy()
        for t_col, s_col in self.predictors_map.items():
            null_tcol_mask = X[t_col].isnull()
            impute_map = self.imputer_maps[t_col]
            for v in impute_map:
                res.loc[null_tcol_mask & (X[s_col] == v), t_col] = impute_map[v]
        return res


class TargetEncoder(TransformerMixin, BaseEstimator):
    def __init__(self, cols=None):
        self.cols = cols
        self.source_allowed_types = ['object']
        self.target_allowed_types = ['int', 'int64', 'float', 'float64']
        self.encoding_maps = {}

    def fit(self, X, y=None, cols=None):
        self.cols = cols if cols is not None else self.cols
        if self.cols is not None:
            X = X[self.cols]

        source_allowed_cols = set(X.select_dtypes(include=self.source_allowed_types).columns)

        assert set(self.cols).issubset(source_allowed_cols)
        assert y.dtype in self.target_allowed_types

        for col in self.cols:
            encode_map = {}
            values = X[col].unique()
            for v in values:
                encode_map[v] = y.loc[X[col] == v].mean()
            self.encoding_maps[col] = encode_map
        return self

    def transform(self, X):
        assert set(self.cols).issubset(set(X.columns))
        res = X.copy()
        for col in self.cols:
            res[col] = res[col].map(self.encoding_maps[col], na_action='ignore')
            res[col][res[col].isnull()] = np.mean(list(self.encoding_maps[col].values()))
        return res

# Early Preparations
data = pd.read_csv(data_path, names=get_features_from_names(names_path))
data = data.replace({missing_val: np.nan})
data = data.dropna(axis=0, subset=[target_col])
numeric_features = get_numeric_from_names(names_path)
data = data.astype(dtype={col: 'float' for col in numeric_features})

# EDA
cols_with_missing = data.columns[data.isnull().any()]
# print("Cols with missing vals: %s" % list(data[cols_with_missing]))
"""Missing data doesn't seem to have semantic meaning in any of the features with missing data"""

listwise_del_cols = []
listwise_normed_avg_thresh = 0.1
for col in cols_with_missing:
    target_avg_missing = data[target_col][data[col].isnull()].mean()
    target_avg_existing = data[target_col][~data[col].isnull()].mean()
    total_avg = data[target_col].mean()
    norm_avg = ((target_avg_existing - target_avg_missing) / total_avg) * (
            data[col].isnull().sum() / (len(data[col]) - data[col].isnull().sum()))
    if np.abs(norm_avg) <= listwise_normed_avg_thresh:
        listwise_del_cols.append(col)

"""All features expect normalized-losses present low normed avg difference
(horsepower,peak-rpm) and (bore,stroke) are missing together, most likely they are MCAR
normalized-losses present extremely high bias in avg price.
Listwise delete all except that one, and look for correlation with some other feature,
to understand whether it is MAR"""

del_listwise = data.dropna(subset=listwise_del_cols, how='any')
del_listwise[['normalized-losses', 'price']] = del_listwise[['normalized-losses', 'price']] / del_listwise[
    ['normalized-losses', 'price']].mean()
a1 = del_listwise.boxplot(column=['price'], by='make', return_type='axes',
                          boxprops=dict(linestyle='-', linewidth=4, color='b'))
a2 = del_listwise.boxplot(column=['normalized-losses'], by='make', ax=a1,
                          boxprops=dict(linestyle='-', linewidth=4, color='r'))
# plt.show()

"""Losses are vague for expensive brands, and pretty solid on others.
Perform mean imputation with respect to the firm only
For firms with no mean losses at all, just impute with the maximal average loss,
as we assume missing losses are for expensive ones"""

# Set one-hot label encoding and target encoding features, depending on unique threshold
nuniques = data.nunique()
ohe_unique_thresh = 5
cat_features = [feature for feature in data.columns if
                getattr(data, feature).dtype == 'object' and feature not in listwise_del_cols]
ohe_features = [feature for feature in cat_features if nuniques[feature] <= ohe_unique_thresh]
target_encoding_features = list(filter(lambda x: x not in ohe_features, cat_features))

# Shuffle the data and split to X,y
data = data.sample(frac=1)
X = data.drop(columns=[target_col])
y = data[target_col]

# Split data to test and train, we'll cross-validate so we don't split for validation set
train_size = 0.85
valid_size = 0.2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 - train_size, random_state=37)

# Pre-processing
ohe = ColumnTransformer(
    transformers=[('onehot', Pipeline(steps=[('ohe', OneHotEncoder(handle_unknown='ignore'))]),
                   ohe_features)],
)

preprocessing_pipeline = Pipeline(steps=[
    ('listwise-del', FeatureDropper(cols=listwise_del_cols)),
    ('average-imp', AverageImputer(predictors_map={'normalized-losses': 'make'})),
    ('target_encoding', TargetEncoder(cols=target_encoding_features)),
    ('preprocessor', ohe),
])

# ohe_data = pd.DataFrame(ohe_transformer.fit_transform(X=data),
#                         columns=ohe_transformer.get_feature_names(),
#                         index=data.index)
# data = pd.concat([data, ohe_data], axis=1)
# data = data.drop(columns=ohe_features)

X = preprocessing_pipeline.fit_transform(X_train, y_train)

my_base_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessing_pipeline),
    ('model', RandomForestRegressor(n_estimators=50))
])

# curr_score = cross_val_score(X=X_train, y=y_train, estimator=my_base_pipeline, cv=int(1 / valid_size)).mean()

# base_pipeline.fit(X_train, y_train)
# base_pipeline.p
