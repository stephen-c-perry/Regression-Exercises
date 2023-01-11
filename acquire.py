import os
import pandas as pd
import env

def pull_telco_data():
    sql_query = """
                select * from customers
                join contract_types using (contract_type_id)
                join internet_service_types using (internet_service_type_id)
                join payment_types using (payment_type_id)
                """
    df = pd.read_sql(sql_query, env.get_connection('telco_churn'))
    return df

def get_telco_data():
    if os.path.isfile('telco.csv'):
        df = pd.read_csv('telco.csv', index_col=0)
    else:
        df = pull_telco_data()
        df.to_csv('telco.csv')
    return df