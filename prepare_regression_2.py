import numpy as np
import pandas as pd
from scipy import stats
import sklearn.preprocessing as p
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import wrangle as w
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import TweedieRegressor




#minmaxscaler
def scale_zillow(train, validate, test):

    train_1 = train.copy()
    validate_1 = validate.copy()
    test_1 = test.copy()

    scale_cols = ['bedrooms', 'bathrooms', 'total_sqft', 'year_built']
    minmax_scaler = p.MinMaxScaler()
    minmax_scaler.fit(train_1[scale_cols])

    train_1[scale_cols] = minmax_scaler.transform(train[scale_cols])
    validate_1[scale_cols] = minmax_scaler.transform(validate[scale_cols])
    test_1[scale_cols] = minmax_scaler.transform(test[scale_cols])

    df_train_1 = pd.DataFrame(train_1).set_index([train_1.index.values])
    df_validate_1 = pd.DataFrame(validate_1).set_index([validate_1.index.values])
    df_test_1 = pd.DataFrame(test_1).set_index([test_1.index.values])

    return df_train_1, df_validate_1, df_test_1




def scale_dataframes(df1, df2, df3):
    scaler = MinMaxScaler()
    df1_scaled = scaler.fit_transform(df1)
    df2_scaled = scaler.transform(df2)
    df3_scaled = scaler.transform(df3)
    return pd.DataFrame(df1_scaled), pd.DataFrame(df2_scaled), pd.DataFrame(df3_scaled)
