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






#minmaxscaler
def scale_zillow(train, validate, test):

    train_1 = train.copy()
    validate_1 = validate.copy()
    test_1 = test.copy()

    scale_cols = ['bedrooms', 'bathrooms', 'area', 'year_built', 'taxamount']
    minmax_scaler = p.MinMaxScaler()
    minmax_scaler.fit(train_1[scale_cols])

    train_1[scale_cols] = minmax_scaler.transform(train[scale_cols])
    validate_1[scale_cols] = minmax_scaler.transform(validate[scale_cols])
    test_1[scale_cols] = minmax_scaler.transform(test[scale_cols])

    df_train_1 = pd.DataFrame(train_1).set_index([train.index.values])
    df_validate_1 = pd.DataFrame(validate_1).set_index([validate.index.values])
    df_test_1 = pd.DataFrame(test_1).set_index([test.index.values])

    return df_train_1, df_validate_1, df_test_1
