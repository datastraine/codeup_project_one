import pandas as pd
import env
import os

def get_connection(db, user=env.user, host=env.host, password=env.password):
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

def get_telco():
    filename = "telco.csv"

    if os.path.isfile(filename):
        return pd.read_csv(filename)
    else:
        # read the SQL query into a dataframe
        df = pd.read_sql("""select * from telco_churn.customers left join telco_churn.internet_service_types
        on telco_churn.internet_service_types.internet_service_type_id = telco_churn.customers.internet_service_type_id left join telco_churn.payment_types
on telco_churn.payment_types.payment_type_id = telco_churn.customers.payment_type_id left join telco_churn.contract_types
on telco_churn.contract_types.contract_type_id = telco_churn.customers.contract_type_id
""", get_connection('telco_churn'))

        # Write that dataframe to disk for later. Called "caching" the data for later.
        df.to_csv(filename, index=False)

        # Return the dataframe to the calling code
        return df 
