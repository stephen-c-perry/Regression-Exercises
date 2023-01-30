import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wrangle as w
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from scipy.stats import linregress
from sklearn.metrics import explained_variance_score
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import TweedieRegressor

import prepare_regression as pr


train, validate, test = w.wrangle_zillow()



def model_prep(df1, df2, df3):
    df1, df2, df3 = county_dummies_all(df1, df2, df3)
    #df1 = df1.drop(columns= ['parcel_id', 'property_id', 'zip_code'])
    #df2 = df2.drop(columns= ['parcel_id', 'property_id', 'zip_code'])
    #df3 = df3.drop(columns= ['parcel_id', 'property_id', 'zip_code'])
    return df1, df2, df3

def X_train_y_train_split(df):
    X_train = df.drop(columns = 'tax_value')
    y_train = df.drop(columns = ['bathrooms' , 'bedrooms' , 'year_built' , 'total_sqft' , 'Los_Angeles' , 'Orange' , 'Ventura'])
    return X_train, y_train


#different function for getting dummies that takes in 3 arguments

def county_dummies_all(train_1, validate_1, test_1):
    train_1, validate_1, test_1 = w.wrangle_zillow()
    train_1_encoded = pd.get_dummies(train_1['county'], drop_first=False)  
    train_1_encoded = train_1.merge(train_1_encoded, left_index=True, right_index=True)
    train_1_encoded = train_1_encoded.drop(columns= 'county')

    validate_1_encoded = pd.get_dummies(validate_1['county'], drop_first=False)  
    validate_1_encoded = validate_1.merge(validate_1_encoded, left_index=True, right_index=True)
    validate_1_encoded = validate_1_encoded.drop(columns= 'county')

    test_1_encoded = pd.get_dummies(test_1['county'], drop_first=False)  
    test_1_encoded = test_1.merge(test_1_encoded, left_index=True, right_index=True)
    test_1_encoded = test_1_encoded.drop(columns= 'county')

    return train_1_encoded, validate_1_encoded, test_1_encoded




#getting county dummies and dropping columns
train_model, validate_model, test_model = model_prep(train, validate, test)

#separating target variable
X_train, y_train = X_train_y_train_split(train_model)
X_validate, y_validate = X_train_y_train_split(validate_model)
X_test, y_test = X_train_y_train_split(test_model)

#scaling
X_train, X_validate, X_test = pr.scale_dataframes(X_train, X_validate, X_test)



y_train['value_pred_mean'] = 527866.30
y_validate['value_pred_mean'] = 527866.30

y_train['value_pred_median'] = 376866.00
y_validate['value_pred_median'] = 376866.00



def GLM(power, alpha):
    # create the model object
    glm = TweedieRegressor(power=power, alpha=alpha)

    # fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series! 
    glm.fit(X_train, y_train.tax_value)

    # predict train
    y_train['value_pred_lm'] = glm.predict(X_train)

    # evaluate: rmse
    rmse_train = rmse(y_train.tax_value, y_train.value_pred_mean)

    # predict validate
    y_validate['value_pred_lm'] = glm.predict(X_validate)

    # evaluate: rmse
    rmse_validate = rmse(y_validate.tax_value, y_validate.value_pred_median)

    return print("RMSE for GLM using TweedieRegressor\nTraining/In-Sample: ", round(rmse_train), 
      "\nValidation/Out-of-Sample: ", round(rmse_validate))





# Sum of Squared Errors SSE
def sse(y_true, y_pred):
    sse = mean_squared_error(y_true, y_pred) * len(y_true)
    return sse


# Mean Squared Error MSE
def mse(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return mse


#Root Mean Squared Error RMSE
def rmse(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    return rmse


# Explained Sum of Squares ESS
def ess(y_true, y_pred):
    mean_y = np.mean(y_true)
    ess = np.sum((y_pred - mean_y)**2)
    return ess


# Total Sum of Squares TSS
def total_sum_of_squares(arr):
    return np.sum(np.square(arr))


# R-Squared R2

def calculate_r2(y_true, y_pred):
    return r2_score(y_true, y_pred)


# Linear Regressions
'''
Quickly calculate r value, p value, and standard error
'''
def linear_regression(x, y):
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    return slope, intercept, r_value, p_value, std_err



#scale