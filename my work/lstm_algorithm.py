import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import LSTMV1
from datetime import date
import yfinance as yf

START = "2010-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

# Define a function to load the dataset

def load_data(ticker):
    data = yf.download(ticker, start=START, end=TODAY)
    data.reset_index(inplace=True)
    return data

data = load_data("VOO")
print(data.head())

data.to_csv(r"C:/Users/rs8c8bh/SP500StockPrc.csv")