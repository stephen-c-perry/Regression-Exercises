import os
import pandas as pd
import env

def sql_zillow_data():
    sql_query = """
                Select bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, taxvaluedollarcnt, yearbuilt, taxamount, fips, propertylandusetypeid
                from properties_2017
                join propertylandusetype USING(propertylandusetypeid)
                where propertylandusetypeid = 261;
                """
    df = pd.read_sql(sql_query, env.get_connection('zillow'))
    return df

def get_zillow_data():
    if os.path.isfile('zillow.csv'):
        df = pd.read_csv('zillow.csv', index_col=0)
    else:
        df = sql_zillow_data()
        df.to_csv('zillow.csv')
    return df