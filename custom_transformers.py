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


# class ListwiseDelete(TransformerMixin):
#
#     def __init__(self, cols):
#         self.cols = cols
#
#     def fit(self, X=None, y=None):
#         rows_to_drop = X[X[self.cols].isnull().any(axis=1)].index
#         X.drop(rows_to_drop, axis='index', inplace=True), y.drop(rows_to_drop, axis='index', inplace=True)
#         return self
#
#     def transform(self, X, y=None):
#         return X


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
    """ The Imputer recieves a @predictor_map that maps between numerical to categorical cols
    For each pair of features <imputed_col, class_col> we impute each missing value in @imputed_col with the mean
    value observed in it for all non-missing entries that hold the same class in the corresponding entry of @class_col
    We build a map for each feature pair, all held under @imputer_maps.
    """

    def __init__(self, predictors_map=None, default_value_func=np.mean):
        self.predictors_map = predictors_map
        self.class_allowed_types = ['object']
        self.imputed_allowed_types = ['int', 'int64', 'float', 'float64']
        self.imputer_maps = {}
        self.default_value_func = default_value_func

    def fit(self, X, y=None, predictors_map=None):
        self.predictors_map = predictors_map if predictors_map is not None else self.predictors_map
        features_pairs = list(self.predictors_map.items())
        class_cols = list(map(lambda x: x[1], features_pairs))
        imputed_cols = list(map(lambda x: x[0], features_pairs))
        source_allowed_cols = set(X.select_dtypes(include=self.class_allowed_types).columns)
        target_allowed_cols = set(X.select_dtypes(include=self.imputed_allowed_types).columns)

        assert set(class_cols).issubset(source_allowed_cols)
        assert set(imputed_cols).issubset(target_allowed_cols)

        for target_col in imputed_cols:
            impute_map = {}

            class_col = self.predictors_map[target_col]
            values = X[class_col].unique()
            for v in values:
                impute_map[v] = X.loc[X[target_col].notnull() & (X[class_col] == v), target_col].mean()

            default_fill_value = self.default_value_func([val for val in impute_map.values() if val is not np.nan])
            for v in values:
                if impute_map[v] is np.nan:
                    impute_map[v] = default_fill_value

            self.imputer_maps[target_col] = impute_map

        return self

    def transform(self, X, y=None):
        res = X.copy()
        for imputed_col, class_col in self.predictors_map.items():
            null_tcol_mask = X[imputed_col].isnull()
            impute_map = self.imputer_maps[imputed_col]
            for v in impute_map:
                res.loc[null_tcol_mask & (X[class_col] == v), imputed_col] = impute_map[v]

            # fill remaining nans with mean value
            if any(res[imputed_col].isnull()):
                res[imputed_col].fillna(np.mean(list(impute_map.values())), inplace=True)
        return res


class TargetEncoder(TransformerMixin, BaseEstimator):
    """Target Encoder encodes the labels of each of the given features (@cols), by computing the mean target for each
    of its categories.
    """
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
