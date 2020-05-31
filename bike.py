import pandas as pd
import os
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from numpy.core.defchararray import zfill
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import f_regression, SelectKBest, mutual_info_regression
from sklearn.pipeline import Pipeline

from custom_transformers import FeatureDropper, RemoveDateTimes, FeatureSelector
from utils import target_means_per_feature, target_std_per_feature, calc_correlations_with_target

data_folder = "Bike-Sharing-Dataset"

date_col = 'dteday'
hour_col = 'hr'
full_date_col = 'fulldt'
target_col = 'cnt'

plot_plots = False

# Load data
day = pd.read_csv(os.path.join(data_folder, 'day.csv'))
hour = pd.read_csv(os.path.join(data_folder, 'hour.csv'))
datasets = [day, hour]

# EDA #
for ds in datasets:
    ds[date_col] = pd.to_datetime(ds[date_col])

# Examine differences between daily and hourly data
hour.insert(loc=len(hour.columns), column=full_date_col,
            value=pd.to_datetime(
                hour[date_col].astype(str) + '-' + hour[hour_col].astype(str).apply(zfill,
                                                                                    width=2),
                format='%Y-%m-%d-%H'))

day_to_target = day.set_index(date_col)[target_col]
aggregated_hour_to_target = hour.resample('D', on=full_date_col)[target_col].sum()
print(all((day_to_target == aggregated_hour_to_target).values))
# Both show identical target values, we conclude that hourly data is just a decomposed version of the daily data.
# We keep on working with daily dataset only
data = day

# Find pearson coefficients between features and target - all features except dteday are numeric
correlations = calc_correlations_with_target(data, target_col, show=plot_plots)

if plot_plots:
    sns.pairplot(data, kind='scatter')
    plt.show()

all_features = data.columns
target_stds = target_std_per_feature(data, target_col, all_features, show=plot_plots)

# 'registered', 'casual' are irrelevant as they cause target leakage - they won't be available in prediction time.
# 'instant' is the record id which correlates well to target
# (probably due to ever-increasing number of rents during time) but also won't be available in prediction time.
leakage_columns = ['registered', 'casual', 'instant']

redundant_columns = []

calc_correlations_with_target(data, target_col='atemp', feature_cols=['temp'], show=plot_plots)
# 'atemp' and 'temp' are almost perfectly correlated
# 'atemp' was found slightly better correlating to target than 'temp' - so we discard 'temp'
redundant_columns.append('temp')

target_means_per_feature(data, target_col='season', feature_cols=['mnth'], show=plot_plots)
# 'season' is redundant due to 'month' - 'month' is a more detailed version of 'season'.
# 'season' coarsely separates December records to winter and spring,
# while people are usually unaware of official season changes.
redundant_columns.append('season')

irrelevant_columns = leakage_columns + redundant_columns
low_correlation_threshold = 0.1  # features with lower correlation are discarded

# extract features that are neither causing leakage nor redundant, and present reasonable linear correlation to target
relevant_corrs = dict(
    filter(lambda x: x[0] not in irrelevant_columns and x[1] > low_correlation_threshold, correlations.items()))
relevant_features = list(relevant_corrs.keys())
num_relevant_features = len(relevant_features)

# Model Construction #
# data splitting
data_shuffled = data.sample(frac=1)
X = data_shuffled.drop(columns=[target_col])
y = data_shuffled[target_col]

train_size = 0.85
valid_size = 0.1  # 10-fold cross-validation

num_bins = 10
bins = np.linspace(y.min(), y.max() + 1, num_bins)
y_binned = np.digitize(y, bins, right=False)
assert sum(y_binned == num_bins) == 0  # Last bin remains empty
if plot_plots:
    plt.hist(y_binned)
    plt.show()

# Perform stratified splitting to keep the train and test sets balanced
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 - train_size, random_state=37, stratify=y_binned)
print(y_train.mean(), y_test.mean())

# Different feature selections, comparing my manually selected to auto selected using two methods
my_fs = Pipeline(steps=[
    ('select_features', FeatureSelector(relevant_features))
])

freg_fs = Pipeline(steps=[
    ('select_features', SelectKBest(score_func=f_regression, k=num_relevant_features))
])

mi_fs = Pipeline(steps=[
    ('select_features', SelectKBest(score_func=mutual_info_regression, k=num_relevant_features))
])

random_fs = Pipeline(steps=[
    ('select_features', FeatureSelector(n_random=num_relevant_features))
])

feature_selectors = [my_fs, freg_fs, mi_fs, random_fs]

best_fs_pipeline = None
my_fs_pipeline = None
best_fs_score = -np.inf

# Find the best pipeline with respect to other feature-selection strategies (f_regression, mutual_info_regression,
# and random as a sanity check)
for fs in feature_selectors:
    curr_pipeline = Pipeline(steps=[
        ('remove_leakage', FeatureDropper(leakage_columns)),
        ('remove_datetime', RemoveDateTimes()),
        ('select_features', fs),
        ('model', RandomForestRegressor(n_estimators=100))
    ])
    if fs == my_fs:
        my_fs_pipeline = curr_pipeline

    curr_score = cross_val_score(X=X_train, y=y_train, estimator=curr_pipeline, cv=int(1 / valid_size)).mean()
    print("Average cv score: %f" % curr_score)
    if curr_score > best_fs_score:
        best_fs_score = curr_score
        best_fs_pipeline = curr_pipeline

# Test Evaluation #
# Fit the best CV model on the whole training data.
# Should not change the model according to these results
best_fs_pipeline.fit(X_train, y_train)
y_pred = best_fs_pipeline.predict(X_test)
print("Test score (MSE) for best fs %d" % mean_squared_error(y_test, y_pred))

if best_fs_pipeline is not my_fs_pipeline:  # Avoid double train
    my_fs_pipeline.fit(X_train, y_train)
y_pred = my_fs_pipeline.predict(X_test)
print("Test score (MSE) for my fs %d" % mean_squared_error(y_test, y_pred))
