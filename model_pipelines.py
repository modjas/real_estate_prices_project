#Pipeline
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, MinMaxScaler

#Models
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

import numpy as np


def set_columns():
    ordinal_cols = ['elevator', 'condition', 'plot_ownership']
    categorical_cols = ['house_type', 'postal_code']
    numerical_cols = ['sqm', 'year_built', 'floors', 'current_floor', 'floor_ratio']

    return ordinal_cols, categorical_cols, numerical_cols

def set_transformers():
    ordinal_transformer = Pipeline(steps=[('ordinal', OrdinalEncoder())])
    categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore', ))])
    minmax_scaler = Pipeline(steps=[('minmax', MinMaxScaler(feature_range=(0,1)))])

    return ordinal_transformer, categorical_transformer, minmax_scaler

def set_preprocessor():
    ordinal_cols, categorical_cols, numerical_cols = set_columns()
    ordinal_transformer, categorical_transformer, minmax_scaler = set_transformers()

    preprocessor = ColumnTransformer(
    transformers = [
        ('ord', ordinal_transformer, ordinal_cols),
        ('cat', categorical_transformer, categorical_cols),
        ('minmax', minmax_scaler, numerical_cols)],
        remainder='passthrough'
        )

    return preprocessor


def lgbm_pipeline():
    preprocessor = set_preprocessor()
    best_params = {'learning_rate': 0.21544346900318823,
                    'max_depth': 128,
                    'min_data_in_leaf': 2,
                    'num_leaves': 16}
    model = LGBMRegressor(**best_params)

    lgbm_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                    ('model', model)])

    return lgbm_pipeline


def random_forest_pipeline():
    preprocessor = set_preprocessor()
    best_params = {'max_depth':50,
                    'max_features': 'sqrt',
                    'min_samples_split': 5,
                    'n_estimators': 120}
    model = RandomForestRegressor(**best_params)

    random_forest_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                            ('model', model)])

    return random_forest_pipeline

def linear_pipeline():
    preprocessor = set_preprocessor()
    model = LinearRegression()

    linear_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                    ('model', model)])


def lgbm_hyperparameters():
    parameters = {
    'model__num_leaves': [8, 16, 32, 64, 128],
    'model__min_data_in_leaf': [2, 4, 8, 16, 32],
    'model__max_depth': [8, 16, 32, 64, 128, -1],
    'model__learning_rate': np.logspace(-3, 0, 10), 
    }

    return parameters

def random_forest_hyperparameters():
    parameters = {
    'model__n_estimators': [10*i for i in range(1,20)],
    'model__max_features': ['auto', 'sqrt'],
    'model__max_depth': [10*i for i in range(1,10)]+[None],
    'model__min_samples_split': [2, 5, 10],
    'model__min_samples_leaf': [1, 2, 4],
#    'bootstrap': [True, False]
    }

    return parameters