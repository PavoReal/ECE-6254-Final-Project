import numpy as np
import seaborn as sns
import yfinance as yf
import pandas as pd
import os
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# RANDOM FOREST

# features-> different lags. 

# param_grid = {
#     'n_estimators' = [100, 200, 300],
#     'criterion' = []
# }
# Load model- yfinance
def getDataSet(ticker, startDate, endDate, loadFromOnline):
    # load model from yfinance or anything from keras
    # Download the data

    if loadFromOnline:
        try:
            data = yf.download(ticker, start=startDate, end=endDate)
            return data
        except yf.errors.YFRateLimitError as e:
            print(f"Rate limit error: {e}. Retrying after a delay.")
            time.sleep(60) #wait 60 seconds.
            try:
                data = yf.download(ticker, start=start_date, end=end_date)
                print(data.head())
            except Exception as inner_e:
                print(f"Second attempt failed: {inner_e}")
                return None
    else:
        filename = ticker + ".csv"
        filepath = os.path.join("C:\\Users\\rs8c8bh\\New folder\\raw", filename)
        if os.path.isfile(filepath):
            data = pd.read_csv(filepath)
            print(data.head())
            return data

def create_lag(seq_length):

    # return new X and y from lag data
    return
#TO-DO
# seq_length would be the number of prior data points used to predict the new data point
def create_randomforest_model(seq_length):
    # have the x values
    # set the y values

    # creating RF regressor
    rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42, criterion='squared_error', max_depth=None, min_samples_split=2
                                         , min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='sqrt', max_leaf_nodes=None, min_impurity_decrease=0,
                                         bootstrap=True, oob_score=False,n_jobs=None,verbose=0,warm_start=False,ccp_alpha=0.0,max_samples=None)
    
    return rf_regressor

def split_dataset():
    # parse csv into X and y datasets
    # split into test and train
    # only for train dataset, create a "lag" 

    return None


#----------------------------------------------------------------------------------------------------------------
# Define the ticker symbol
ticker = "CEF"

# Define the date range
start_date = "2020-01-01"
end_date = "2023-12-31"
myDataset = getDataSet(ticker,start_date,end_date,False)
#print(myDataset.head())
#save the data to a csv file.
#myDataset.to_csv("MyData_historical_data.csv")
