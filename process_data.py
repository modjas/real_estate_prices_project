#Gathering data
import requests
from bs4 import BeautifulSoup
import pandas as pd

#ML data processing
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, MinMaxScaler

from lightgbm import LGBMRegressor

def parse_website(postal_code):
    '''
    Connects to asuntojen.hintatiedot.fi and pulls data on sold apartments from the last 12 months.
    
    
    Parameters:
    postal_code (str): Postal area code to get data for
    
    Returns:
    Nested list with apartment data
    
    '''
    URL = 'https://asuntojen.hintatiedot.fi/haku/?c=&cr=1&ps=' + postal_code + '&nc=0&amin=&amax=&renderType=renderTypeTable&search=1'
    
    page = requests.get(URL)

    soup = BeautifulSoup(page.content, 'html.parser')

    main_table = soup.find(id='mainTable')

    apartments = main_table.find_all('tr', class_="")
    
    
    local_area_apartments = []
    for a in apartments[4:-2]: # First 4 and last 2 'td' tags are not apartment information
        elements = a.find_all('td')

        parsed_apartment = [element.text.replace("\xa0", "") for element in elements]
        
        if len(parsed_apartment) < 2:
            continue
        parsed_apartment[3] = parsed_apartment[3].replace(",", ".") #Change to international decimal delimiter
        parsed_apartment[10] = parsed_apartment[10].split("\t")[0] #Weird formatting in source website
        parsed_apartment.append(postal_code)
        local_area_apartments.append(parsed_apartment)
        
    return local_area_apartments

def download_apartment_data(filename='apartment_prices.csv'):
    #TODO Include only real postal codes, now contains some non-existant
    parsed_apartments = []
    postal_codes_helsinki = ['00'+str(10*i) for i in range(10,100)] #All Helsinki postal codes
    postal_codes_espoo = ['0'+str(10*i) for i in range(210,298)]
    postal_codes_vantaa = ['0'+str(10*i) for i in range(120,177)]

    postal_codes = postal_codes_helsinki + postal_codes_espoo + postal_codes_vantaa

    #Loop over postal codes and scrape data
    count = 0
    for postal_code in postal_codes:
        apartments = parse_website(postal_code)
        for apartment in apartments:
            parsed_apartments.append(apartment)
        if count%10 == 0:
            print(f"Parsed {count}/{len(postal_codes)} postal codes")
        count += 1
    print("Parsed all postal codes")

    #Convert scraped data into pandas dataframe
    columns = ['neighborhood', 'apartment_type', 'house_type', 'sqm', 'price', 'price_per_sqm', 'year_built', 'floor', 'elevator', 'condition', 'plot_ownership', 'energy_class', 'postal_code']
    df = pd.DataFrame(parsed_apartments,columns=columns)

    to_numeric_columns = ['sqm', 'price', 'price_per_sqm', 'year_built']
    for column in to_numeric_columns:
        df[column] = pd.to_numeric(df[column])

    #Outputting data to .csv file in order to save data and avoid the need to pull data again
    df.to_csv(filename, index=False)

def parse_floor(row):
    '''
    Converts the string describing current and total floors to usable features.
    '''
    floors = row['floor'].split('/')
    try:
        row['floors'] = float(floors[1])
        row['current_floor'] = float(floors[0])
        row['floor_ratio'] = float(floors[0])/float(floors[1])
    except:
        pass
    return row


def preprocess_data(df):
    # We will use house type, sqm, price, year_built, floor, elevator, condition, plot_ownership and postal_code
    # To predict the price_per_sqm
    cols = ['price_per_sqm','house_type', 'sqm', 'price', 'year_built', 'floor', 'elevator', 'condition', 'plot_ownership', 'postal_code']
    df_ml = df[cols].copy(deep=True)
    df_ml.replace('', np.nan, inplace=True) #Replace empty cells with nan for easier use of built-in methods

    #Find columns with missing values
    columns_with_missing_values = [col for col in df_ml.columns if df_ml[col].isna().any()]

    #As the count of rows containing missing values is somewhat low, the corresponding rows are dropped
    df_ml.dropna(axis=0, inplace=True)

    outliers = df_ml[(df_ml['price_per_sqm'] > 30000) | (df_ml['price_per_sqm'] < 400)]
    df_ml.drop(outliers.index, axis=0, inplace=True)

    df_ml['floors'] = pd.Series()
    df_ml['current_floor'] = pd.Series()
    df_ml['floor_ratio'] = pd.Series()

    df_ml = df_ml.apply(parse_floor, axis='columns')

    return df_ml

def load_train_test(df, test_size=0.2):
    features = df.drop(['price_per_sqm', 'floor', 'price'], axis=1)
    labels = df['price_per_sqm']

    return train_test_split(features, labels, random_state=0, test_size=test_size) 


def lgbm_pipeline():
    ordinal_cols = ['elevator', 'condition', 'plot_ownership']
    categorical_cols = ['house_type', 'postal_code']
    numerical_cols = ['sqm', 'year_built', 'floors', 'current_floor', 'floor_ratio']


    ordinal_transformer = Pipeline(steps=[('ordinal', OrdinalEncoder())])
    categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore', ))])
    minmax_scaler = Pipeline(steps=[('minmax', MinMaxScaler(feature_range=(0,1)))])

    preprocessor = ColumnTransformer(
        transformers = [
            ('ord', ordinal_transformer, ordinal_cols),
            ('cat', categorical_transformer, categorical_cols),
            ('minmax', minmax_scaler, numerical_cols)],
            remainder='passthrough'
        )


    lgbm_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                ('model_lgbm', LGBMRegressor())])

    return lgbm_pipeline