import numpy as np
import seaborn as sns
import yfinance as yf
import pandas as pd
import os
import time
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV

# RANDOM FOREST

# features-> different lags. 

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

def create_lag(seq_length, dataset):
    # return new X and y from lag data
    X = np.zeros((len(dataset) - seq_length, seq_length))
    for i in range(seq_length, len(dataset)):
        X[i - seq_length] = dataset[i - seq_length : i].flatten()
    y = dataset[seq_length : ].flatten()
    return X, y

# add GridSearch for parameter tuning
def grid_search(xtrain, ytrain):
    grid_search = GridSearchCV(estimator=create_randomforest_model(), param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', verbose=2, n_jobs=-1)
    grid_search.fit(xtrain, ytrain)
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_
    print(best_params)
    return best_model

# seq_length would be the number of prior data points used to predict the new data point
def create_randomforest_model():
    # creating RF regressor
    rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42, criterion='squared_error', max_depth=None, min_samples_split=2
                                         , min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='sqrt', max_leaf_nodes=None, min_impurity_decrease=0,
                                         bootstrap=True, oob_score=False,n_jobs=None,verbose=0,warm_start=False,ccp_alpha=0.0,max_samples=None)
    return rf_regressor

def train_model(xtrain, ytrain, xtest):

    best_model= grid_search(xtrain, ytrain)
    # predicting the test set
    yPredict = best_model.predict(xtest)
    return yPredict

def model_evaluation(yPred, yCorrect):
    meanSqErr = mean_squared_error(yCorrect, yPred)
    meanAbsErr = mean_absolute_error(yCorrect, yPred)
    return meanSqErr, meanAbsErr

# this is only helpful if we want to include multiple features in one matrix. RF does poorly if it's only one feature
def create_feature_matrix(*args):
    Xmat = []
    for i in range(len(args[0])):
        feature = []
        for arg in args:
            feature.append(arg[i].tolist())
        Xmat.append(np.array(feature).flatten())
    Xmat = np.array(Xmat)
    return Xmat