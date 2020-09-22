import pandas as pd
import numpy as np
import acquire
from sklearn.model_selection import train_test_split

def prep_telco():
    telco = acquire.get_telco()
    # drop duplicate columns from the joins
    telco.drop(columns=['internet_service_type_id.1', 'payment_type_id.1', 'contract_type_id.1', 
            'internet_service_type_id', 'payment_type_id', 'contract_type_id'], inplace = True)
    # Turn the total_charages column into a float column
    telco['total_charges'] = pd.to_numeric(telco['total_charges'],errors='coerce')
    # Create a int bool for churn where churn = 1
    telco.replace({'churn': {'No': 0, 'Yes': 1}}, inplace = True)
    # Creates a tenue_year column 
    telco['tenure_year'] = round(telco['tenure']/12, 2)
    # creates a single_no_dependents column to see if being alone impacts churn 
    telco['single_no_dependents'] = (telco['partner'] == 'No') & (telco['dependents'] == 'No')
    # Crates a multiple phone line column to see if having phone service with multiple phone lines impacts churn
    telco['multiple_phone_lines'] = (telco['phone_service'] == 'Yes') & (telco['multiple_lines'] == 'Yes')
    # Creates a streaming column to look and see if a customer streams at all influences churn
    telco['streaming'] =  (telco['streaming_tv'] == 'Yes') | (telco['streaming_movies'] == 'Yes')
    # Creates a backedup and secured column to see if a customers who are security consicious impact churn
    telco['backedup_and_secured'] = (telco['online_security'] == 'Yes') & (telco['online_backup'] == 'Yes')
    # creates a has internet column to see if having internet at all impacts churn
    telco['has_internet'] = (telco['internet_service_type'] != 'None')
    # Creates a monthly charages column for analysis
    telco['monthly_charges'] = telco['total_charges']/telco['tenure']
    # Create a monthly_charage >= 75 bool value column
    telco['monthly_75+'] = telco['monthly_charges'] >= 75
    # Turns all True and False/Yes and No values into 1s and 0s for easier analysis
    telco = telco.applymap(lambda x: 0 if x == False else x)
    telco = telco.applymap(lambda x: 1 if x == True else x)
    telco = telco.applymap(lambda x: 1 if x == 'Yes' else x)
    telco = telco.applymap(lambda x: 0 if x == 'No' else x)
    telco = telco.applymap(lambda x: 0 if x == 'No internet service' else x)
    telco = telco.applymap(lambda x: 0 if x == 'No phone service' else x)
    # We'll drop the rows where total_charages is blank as these are new customers
    telco.dropna(inplace = True) 
    # Splits the df into train, validate, test sets for analysis
    train_validate, test = train_test_split(telco, test_size=.2, random_state=333, stratify=telco.churn)
    train, validate = train_test_split(train_validate, 
                                       test_size=.25, 
                                       random_state=333, 
                                       stratify=train_validate.churn)
    return train, validate, test, telco