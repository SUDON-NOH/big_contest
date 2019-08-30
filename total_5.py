import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import numpy as np
import random
from sklearn.model_selection import train_test_split
import scipy as cp

class data_analysis:


    def __init__(self):
        print("Import")

    def read_data(self, filename):
        print("Step_1")

        print("read_csv")
        data = pd.read_csv("'" + filename + ".csv'", index_col= 'acc_id')
        print(data.describe())

        print("MinmaxScaler")
        data = data.drop(columns = 'Unnamed: 0')

        data_1 = data[['survival_time', 'amount_spent']]

        data.drop(columns = ['survival_time', 'amount_spent'], inplace = True)
        scaler = MinMaxScaler()
        data[:] = scaler.fit_transform(data[:])
        data = pd.merge(data, data_1, how = 'inner', left_index = True, right_index = True)

        # ======================================================================================================================
    def



