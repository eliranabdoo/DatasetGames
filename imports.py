import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
import re

from custom_transformers import ClassMeanImputer, TargetEncoder, ListwiseDelete

data_path = "./Imports/imports-85.data"
names_path = "./Imports/imports-85.names"

target_col = 'price'
attribute_match_regex = "\d{1,2}\. (.+?):"
missing_val = '?'


def get_features_from_names(path):
    """
    Extract feature names from .names file of 'imports' dataset
    :param path: path to '.names' file
    :return: list of string, representing all features
    """
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


def get_numeric_cols_from_names_file(path):
    """
    Infer which features are numeric from '.names' file of 'imports' dataset
    :param path: path to '.names' file
    :return: list of strings, representing numeric features
    """
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


# Early preparations
data = pd.read_csv(data_path, names=get_features_from_names(names_path))
numeric_features = get_numeric_cols_from_names_file(names_path)

data = data.replace({missing_val: np.nan})  # Should be applied to new data
data = data.dropna(axis=0, subset=[target_col])  # Shouldn't be applied to new data, target doesn't exist
data = data.astype(dtype={col: 'float' for col in numeric_features})  # numeric features are loaded as strings

# EDA #
cols_with_missing_values = data.columns[data.isnull().any()]
print("Cols with missing vals: %s" % str(cols_with_missing_values))
# Missing data doesn't seem to have semantic meaning in any of the features with missing data (No MNAR)

listwise_del_cols = []
listwise_normed_avg_thresh = 0.1
for col in cols_with_missing_values:
    target_avg_missing = data[target_col][data[col].isnull()].mean()
    target_avg_existing = data[target_col][~data[col].isnull()].mean()
    print("Feature %s - missing entries average: %f - existing entries average: %f" % (
        col, target_avg_missing, target_avg_existing))
    target_total_avg = data[target_col].mean()

    # Asses the difference in target average for missing and existing values, weigh by portion of missing values

    normalized_average_difference = (target_avg_existing - target_avg_missing) / target_total_avg
    missing_to_existing_ratio = data[col].isnull().sum() / data[col].notnull().sum()
    missing_importance = normalized_average_difference * missing_to_existing_ratio
    if np.abs(missing_importance) <= listwise_normed_avg_thresh:
        listwise_del_cols.append(col)

# 'normalized-losses' has a large gap between missing and existing entries target average
# the pairs ('bore', 'stroke') and ('horsepower', 'peak-rpm') present the same averages, check if they are missing together
# 'num-of-doors' also has very similar averages to ('horsepower', 'peak-rpm')

print("%d different missing entries" % (sum(data['bore'].isnull() ^ data['stroke'].isnull())))
print("%d different missing entries" % sum(data['horsepower'].isnull() ^ data['peak-rpm'].isnull()))
print("%d different missing entries" % sum(data['horsepower'].isnull() ^ data['num-of-doors'].isnull()))

# the pairs ('bore', 'stroke') and ('horsepower', 'peak-rpm') are indeed missing together
# both present low numbers of missing entries, so as 'num-of-doors' although it's not missing together with ('horsepower', 'peak-rpm')
# check for common features values between missing groups

for feature in ['bore', 'horsepower', 'num-of-doors']:
    entries = data[data[feature].isnull()]
    similar_features = [col for col in data.columns if
                        col != feature and len(set([entry[col] for index, entry in entries.iterrows()])) == 1]
    print("Inspected feature %s has identical feature values in missing entires under features: %s" % (
        feature, str(similar_features)))

"""All features expect normalized-losses present low normed avg difference
(horsepower,peak-rpm) and (bore,stroke) are missing together, most likely they are MCAR
normalized-losses present extremely high bias in avg price.
Listwise delete all except that one, and look for correlation with some other feature,
to understand whether it is an MAR"""

del_listwise = data.dropna(subset=listwise_del_cols, how='any')
del_listwise[['normalized-losses', 'price']] = del_listwise[['normalized-losses', 'price']] / del_listwise[
    ['normalized-losses', 'price']].mean()
a1 = del_listwise.boxplot(column=['price'], by='make', return_type='axes',
                          boxprops=dict(linestyle='-', linewidth=4, color='b'))
a2 = del_listwise.boxplot(column=['normalized-losses'], by='make', ax=a1,
                          boxprops=dict(linestyle='-', linewidth=4, color='r'))
plt.show()

"""Losses are vague for expensive brands ('make' feature), and pretty solid on others.
Perform mean imputation with respect to the firm only
For firms with no mean losses at all, just impute with the maximal average loss,
as we assume missing losses are for expensive makers"""

# Set one-hot label encoding and target encoding features, depending on unique threshold
nuniques = del_listwise.nunique()
ohe_unique_thresh = 5
categorial_features = [feature for feature in data.columns if getattr(data, feature).dtype == 'object']
ohe_features = [feature for feature in categorial_features if nuniques[feature] <= ohe_unique_thresh]
target_encoding_features = list(filter(lambda x: x not in ohe_features, categorial_features))

# Model Construction #
# Shuffle the data and split to X,y
data = data.sample(frac=1)
X = data.drop(columns=[target_col])
y = data[target_col]
numeric_features.remove(target_col)
ListwiseDelete(cols=listwise_del_cols).fit_transform(X, y)

# Split data to test and train, cross-validate on train (valid_size taken on train part)
num_bins = 10
bins = np.linspace(y.min(), y.max() + 1, num_bins)
y_binned = np.digitize(y, bins, right=False)

train_size = 0.85
valid_size = 0.1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 - train_size, random_state=37, stratify=y_binned)
print(y_train.mean(), y_test.mean())

# Pre-processing pipeline
ohe = ColumnTransformer(
    transformers=[('onehot', OneHotEncoder(handle_unknown='ignore'), ohe_features)], remainder='passthrough')

preprocessing_pipeline = Pipeline(steps=[
    ('average-imp', ClassMeanImputer(predictors_map={'normalized-losses': 'make'})),
    ('target-encoding', TargetEncoder(cols=target_encoding_features)),
    ('one-hot-encoding', ohe),
])

my_pipeline = Pipeline(steps=[
    ('pre-process', preprocessing_pipeline),
    ('model', RandomForestRegressor(n_estimators=100))
])

pipelines = {'my_pipeline': my_pipeline}

# Simple pipelines with different imputation strategies
imputation_strategies = ['mean', 'median', 'most_frequent']
for st in imputation_strategies:
    impute_pipeline = Pipeline(steps=[
        ('impute', SimpleImputer(strategy=st, missing_values=np.nan))
    ])

    encode_pipeline = Pipeline(steps=[
        ('encode', OneHotEncoder(handle_unknown='ignore'))
    ])

    impute_and_encode = ColumnTransformer(
        transformers=[('mean-imputer', impute_pipeline, numeric_features),
                      ('ohe', encode_pipeline, categorial_features)])

    generic_pipeline = Pipeline(steps=[
        ('preprocess', impute_and_encode),
        ('model', RandomForestRegressor(n_estimators=100))
    ])
    pipelines['%s_pipeline' % st] = generic_pipeline

best_pipeline_name = None
best_score = -np.inf

for pl_name, pl in pipelines.items():
    curr_score = cross_val_score(X=X_train, y=y_train, estimator=pl, cv=int(1 / valid_size)).mean()
    print("CV score for %s is: %f" % (pl_name, curr_score))

    if curr_score > best_score:
        best_pipeline_name = pl_name
        best_score = curr_score
best_pipeline = pipelines[best_pipeline_name]

# Test Evaluation #
best_pipeline.fit(X_train, y_train)
y_pred = best_pipeline.predict(X_test)
print("Test score (MSE) for best pipeline: %d" % mean_squared_error(y_test, y_pred))

if best_pipeline is not my_pipeline:  # Avoid double train
    my_pipeline.fit(X_train, y_train)
y_pred = my_pipeline.predict(X_test)
print("Test score (MSE) for my_pipeline %d" % mean_squared_error(y_test, y_pred))
