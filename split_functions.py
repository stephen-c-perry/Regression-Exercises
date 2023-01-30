from sklearn.model_selection import train_test_split


#splits data into train, validate, and test
def split_data(df):
    train, test = train_test_split(df, train_size = 0.8, random_state = 41)
    train, validate = train_test_split(train, train_size = 0.7, random_state = 41)
    return train, validate, test


#seperates target variable from rest of data
def X_y_split(df, target):
    train, validate, test = split_data(df)
    
    X_train = train.drop(columns = target)
    y_train = train[target]

    X_validate = validate.drop(columns = target)
    y_validate = validate[target]

    X_test = test.drop(columns = target)
    y_test = test[target]

    return X_train, y_train, X_validate, y_validate, X_test, y_test


#groups discrete variables into list by having object data type
def discrete_columns(df):
    cat_cols= df.select_dtypes(include=['object']).columns.tolist()
    return cat_cols

#groups continuous variables into list by not having object data type
def continuous_columns(df):
    num_cols= df.select_dtypes(exclude=['object']).columns.tolist()
    return num_cols