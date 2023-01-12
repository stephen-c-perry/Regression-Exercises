import os
import env
import acquire
import pandas as pd
from sklearn.model_selection import train_test_split




def get_zillow():

    if os.path.isfile('zillow.csv'):
        df = pd.read_csv('zillow.csv', index_col=0)

    else:
        df = get_zillow_data()
        df.to_csv('zillow.csv')

    return df





def prep_zillow(df):

    df = df.rename(columns={'bedroomcnt':'bedrooms', 
                          'bathroomcnt':'bathrooms', 
                          'calculatedfinishedsquarefeet':'area',
                          'taxvaluedollarcnt':'tax_value', 
                          'yearbuilt':'year_built'})

    df = df.dropna()
    df.drop_duplicates(inplace=True)
    
    train_validate, test = train_test_split(df, test_size=.2, random_state=123)
    train, validate = train_test_split(train_validate, test_size=.3, random_state=123)
    
    return train, validate, test


def wrangle_zillow():

    train, validate, test = prep_zillow(get_zillow())
    
    return train, validate, test