import env
import acquire
import pandas as pd
from sklearn.model_selection import train_test_split


def prep_telco(df):
    df['total_charges'] = (df.total_charges + '0').astype('float')
    df = df.drop(columns = ['internet_service_type_id', 'contract_type_id', 'payment_type_id'])
    cat_cols = df.drop(columns = 'customer_id').select_dtypes(include=object).columns.to_list()
    cat_col_bin = []
    for i in range(len(cat_cols)):
        if len(df[cat_cols[i]].value_counts()) == 2:
            cat_col_bin.append(cat_cols[i])
    bin_dummies = pd.get_dummies(df[cat_col_bin],drop_first=True)
    not_bin = []
    for i in range(len(cat_cols)):
        if len(df[cat_cols[i]].value_counts()) > 2:
            not_bin.append(cat_cols[i])
    not_bin = pd.get_dummies(df[not_bin],drop_first=True)
    df.drop(['gender', 'partner', 'dependents', 'phone_service','multiple_lines', 'online_security', 'online_backup','device_protection', 'tech_support','streaming_tv', 'streaming_movies','paperless_billing','churn', 'contract_type', 'internet_service_type', 'payment_type'], axis =  1, inplace=True)

    df = pd.concat([df, bin_dummies], axis=1)
    df = pd.concat([df, not_bin], axis=1)
    return df



# Split Data (train, validate, test)
'''
split_data is a function that takes in a dataframe splits that data first into train_validate and 
test dataframes, then splits the train_validate set further into train and validate.  This results in 3 dataframes to use for modeling.
'''

def split_data(df):
    '''
    take in a DataFrame and return train, validate, and test DataFrames; stratify on churn_Yes.
    return train, validate, test DataFrames.
    '''
    train_validate, test = train_test_split(df, test_size=.2, random_state=123, stratify=df.churn_Yes)
    train, validate = train_test_split(train_validate, 
                                       test_size=.3, 
                                       random_state=123, 
                                       stratify=train_validate.churn_Yes)
    return train, validate, test
