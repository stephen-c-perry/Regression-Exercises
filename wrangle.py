import os
import env
import acquire
import pandas as pd
from sklearn.model_selection import train_test_split




def get_zillow():

    if os.path.isfile('zillow_project.csv'):
        df = pd.read_csv('zillow_project.csv', index_col=0)

    else:
        df = acquire.sql_zillow_data()
        df.to_csv('zillow_project.csv')

    return df







def prep_zillow(df):
    #rename columns
    df = df.rename(columns={'bedroomcnt':'bedrooms', 
                          'bathroomcnt':'bathrooms',
                          'yearbuilt':'year_built',
                          'calculatedfinishedsquarefeet':'total_sqft',
                          'fips': 'county',
                          'taxvaluedollarcnt': 'tax_value'
                          })
    #set upper and lower limits for bathrooms and bedrooms
    df = df[df.bathrooms <= 8]
    df = df[df.bathrooms >= 1]
    df = df[df.bedrooms <= 8]
    df = df[df.bedrooms >= 2]
    
     #convert data types 
    
    df.total_sqft = df.total_sqft.convert_dtypes(int)
    df.bedrooms = df.bedrooms.convert_dtypes(int)
    df.year_built = df.year_built.convert_dtypes(int)

     #replace 4 digit fips code with name of county
    df.county = df.county.replace(6059.0,'Orange').replace(6037.0,'Los_Angeles').replace(6111.0,'Ventura')
    

    df.drop_duplicates(inplace=True)
    
    train_validate, test = train_test_split(df, test_size=.2, random_state=123)
    train, validate = train_test_split(train_validate, test_size=.3, random_state=123)
    
    return train, validate, test


#wrangle_zillow combines the acquire and prep functions into one function
def wrangle_zillow():

    train, validate, test = prep_zillow(get_zillow())
    
    return train, validate, test