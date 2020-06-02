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

date_col = "dteday"
hour_col = "hr"
full_date_col = "fulldt"
target_col = "cnt"

PLOTS = False

# Load data
day = pd.read_csv(os.path.join(data_folder, "day.csv"))
hour = pd.read_csv(os.path.join(data_folder, "hour.csv"))
datasets = [day, hour]

#########
## EDA ##
#########

for ds in datasets:
    ds[date_col] = pd.to_datetime(ds[date_col])

# Examine differences between daily and hourly data
hour.insert(loc=len(hour.columns), column=full_date_col,
            value=pd.to_datetime(
                hour[date_col].astype(str) + "-" + hour[hour_col].astype(str).apply(zfill,
                                                                                    width=2),
                format="%Y-%m-%d-%H"))

day_to_target = day.set_index(date_col)[target_col]
aggregated_hour_to_target = hour.resample("D", on=full_date_col)[target_col].sum()
print(all((day_to_target == aggregated_hour_to_target).values))
# Both show identical target values, we conclude that hourly data is a decomposed version of the daily data,
# which encompasses more data.
# We keep on working with hourly dataset only
data = hour

# Find pearson coefficients between features and target - all features except dteday are of numeric datatype, although
# some are categorical encoded in a semantic way (i.e. months are encoded 1 to 12)
correlations = calc_correlations_with_target(data, target_col, show=PLOTS)

# "registered", "casual" are irrelevant as they cause target leakage - they won"t be available in prediction time.
# "instant" is the record id which correlates well to target
# (probably due to ever-increasing number of rents during time) but also won"t be available in prediction time.
leakage_columns = ["registered", "casual"]
del correlations["registered"]
del correlations["casual"]

redundant_columns = []
top_5 = list(map(lambda x: x[0], sorted(tuple(correlations.items()), key=lambda x: x[1])[-5:]))
if PLOTS:
    sns.pairplot(data[top_5], kind="scatter")
    plt.show()
print(calc_correlations_with_target(data, target_col="atemp", feature_cols=["temp"], show=False)["temp"])
# "atemp" and "temp" are almost perfectly correlated
redundant_columns.append("temp")

weather_target_std = target_std_per_feature(data, target_col, ["weathersit", "season"], show=PLOTS)
redundant_columns.append("season")
redundant_columns.append("windspeed")

target_means_per_feature(data, target_col, feature_cols=["hr", "mnth", "yr"], sort=False, show=PLOTS)

redundant_columns.append("yr")

irrelevant_columns = leakage_columns + redundant_columns
low_correlation_threshold = 0.05  # features with lower correlation are discarded

relevant_corrs = dict(
    filter(lambda x: x[0] not in irrelevant_columns and x[1] > low_correlation_threshold, correlations.items()))
relevant_features = list(relevant_corrs.keys())
print("Chosen features are: %s" % list(relevant_features))
num_relevant_features = len(relevant_features)

########################
## Model Construction ##
########################

# we shuffle the data as it is ordered by time - although it is not required due to the stratified split
data_shuffled = data.sample(frac=1)
X = data_shuffled.drop(columns=[target_col])
y = data_shuffled[target_col]

train_size = 0.7
valid_size = 0.1  # 10-fold cross-validation

num_bins = 10
bins = np.linspace(y.min(), y.max() + 1, num_bins)
y_binned = np.digitize(y, bins, right=False)
assert sum(y_binned == num_bins) == 0  # Last bin remains empty
if PLOTS:
    plt.hist(y_binned)
    plt.show()

# Perform stratified splitting to keep the train and test sets balanced
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 - train_size, random_state=37, stratify=y_binned)
print(y_train.mean(), y_test.mean())

# Different feature selections, comparing my manually selected to auto selected using two methods
my_fs = Pipeline(steps=[
    ("select_features", FeatureSelector(relevant_features))
])

freg_fs = Pipeline(steps=[
    ("select_features", SelectKBest(score_func=f_regression, k=num_relevant_features))
])

mi_fs = Pipeline(steps=[
    ("select_features", SelectKBest(score_func=mutual_info_regression, k=num_relevant_features))
])

random_fs = Pipeline(steps=[
    ("select_features", FeatureSelector(n_random=num_relevant_features))
])

feature_selectors = {'my_fs': my_fs,
                     'fregression_fs': freg_fs,
                     'mutual_info_fs': mi_fs,
                     'random_fs': random_fs}
pipelines = {}

for fs_name, fs in feature_selectors.items():
    curr_pipeline = Pipeline(steps=[
        ("remove_leakage", FeatureDropper(leakage_columns)),
        ("remove_datetime", RemoveDateTimes()),
        ("select_features", fs),
        ("model", RandomForestRegressor(n_estimators=100))
    ])
    if fs == my_fs:
        my_fs_pipeline = curr_pipeline
    pipelines[fs_name] = curr_pipeline

best_fs_pipeline = None
best_fs_pipeline_name = None
my_fs_pipeline = None
best_fs_score = -np.inf

######################
## Model Evaluation ##
######################

# Find the best pipeline with respect to other feature-selection strategies (f_regression, mutual_info_regression,
# and random as a sanity check)
for pl_name, pl in pipelines.items():
    curr_score = cross_val_score(X=X_train, y=y_train, estimator=pl, cv=int(1 / valid_size)).mean()
    print("Average cv score: %f" % curr_score)
    if curr_score > best_fs_score:
        best_fs_score = curr_score
        best_fs_pipeline_name = pl_name

################
## Test Score ##
################

# Fit the best CV model on the whole training data.
# Should not change the model according to these results
best_fs_pipeline = pipelines[best_fs_pipeline_name]
best_fs_pipeline.fit(X_train, y_train)
y_pred = best_fs_pipeline.predict(X_test)
print("Test score (MSE) for best fs %s: %d" % (best_fs_pipeline_name, mean_squared_error(y_test, y_pred)))

# MSE for all other models
for pl_name, pl in pipelines.items():
    if pl != best_fs_pipeline:  # Avoid double train
        pl.fit(X_train, y_train)
        y_pred = pl.predict(X_test)
        print("Test score (MSE) for %s: %d" % (pl_name, mean_squared_error(y_test, y_pred)))
