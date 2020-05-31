from sklearn.base import TransformerMixin, BaseEstimator
import numpy as np
from sklearn.impute import SimpleImputer
import pandas as pd

""" Contains several custom trnsformers for data imputation and label encodings
'fit' and 'transform' are parameterized with X and y for compatibility reasons with sklearn's pipelines
"""


class FeatureDropper(TransformerMixin):

    def __init__(self, cols):
        self.cols = cols

    def fit(self, X=None, y=None):
        return self

    def transform(self, X, y=None):
        return X.drop(columns=self.cols)


class ListwiseDelete(TransformerMixin):

    def __init__(self, cols):
        self.cols = cols

    def fit(self, X=None, y=None):
        rows_to_drop = X[X[self.cols].isnull().any(axis=1)].index
        X.drop(rows_to_drop, axis='index', inplace=True), y.drop(rows_to_drop, axis='index', inplace=True)
        return self

    def transform(self, X, y=None):
        return X


class FeatureSelector(TransformerMixin):

    def __init__(self, cols=None, n_random=None):
        if cols is not None:
            self.cols = cols
        elif n_random is None:
            self.cols = slice(None, None)
        else:
            self.cols = None
            self.n_random = n_random

    def fit(self, X=None, y=None):
        return self

    def transform(self, X, y=None):
        if self.cols is None:
            self.cols = np.random.choice(X.columns, self.n_random, replace=False)
        return X[self.cols]


class RemoveDateTimes(TransformerMixin):

    def fit(self, X=None, y=None):
        return self

    def transform(self, X, y=None):
        return X.drop(columns=list(X.select_dtypes(include=['datetime64'])))


class ClassMeanImputer(TransformerMixin, BaseEstimator):
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

        for target_col in target_cols:
            impute_map = {}

            source_col = self.predictors_map[target_col]
            values = X[source_col].unique()
            for v in values:
                impute_map[v] = X.loc[X[target_col].notnull() & (X[source_col] == v), target_col].mean()
            self.imputer_maps[target_col] = impute_map

        for impute_map in self.imputer_maps.values():
            default_val = self.default_val_func([val for val in impute_map.values() if val is not np.nan])
            for v, avg in impute_map.items():
                if avg is np.nan:
                    impute_map[v] = default_val

        return self

    def transform(self, X, y=None):
        res = X.copy()
        for target_col, source_col in self.predictors_map.items():
            null_tcol_mask = X[target_col].isnull()
            impute_map = self.imputer_maps[target_col]
            for v in impute_map:
                res.loc[null_tcol_mask & (X[source_col] == v), target_col] = impute_map[v]

            # fill remaining nans with mean value
            if any(res[target_col].isnull()):
                res[target_col].fillna(np.mean(list(impute_map.values())), inplace=True)
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

    def transform(self, X, y=None):
        assert set(self.cols).issubset(set(X.columns))
        res = X.copy()
        for col in self.cols:
            res[col] = res[col].map(self.encoding_maps[col], na_action='ignore')
            res.loc[res[col].isnull(), col] = np.mean(list(self.encoding_maps[col].values()))
        return res

# class SimpleImputerRetainColumns(TransformerMixin, BaseEstimator):
#     def __init__(self, strategy='mean', missing_values=np.nan):
#         self.imputer = SimpleImputer(strategy, missing_values)
#         self.cols = None
#
#     def fit(self, X, y=None):
#         self.cols = X.columns
#         self.imputer.fit(X, y)
#         return self
#
#     def transform(self, X, y=None):
#         return pd.DataFrame(self.imputer.transform(X), columns=self.cols)
