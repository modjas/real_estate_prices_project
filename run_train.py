import process_data as red
import os
import pandas as pd
import numpy as np
import joblib


#ML models and hyperparameter optimisation
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
from lightgbm import LGBMRegressor

def run_train_test_lgbm():
    lgbm_pipeline = red.lgbm_pipeline()

    lgbm_parameters_model_complexity = {
    'model_lgbm__num_leaves': [8, 16, 32, 64, 128],
    'model_lgbm__min_data_in_leaf': [2, 4, 8, 16, 32],
    'model_lgbm__max_depth': [8, 16, 32, 64, 128, -1],
    }

    lgbm_parameters_training_speed = {
    'model_lgbm__learning_rate': np.logspace(-3, 0, 10), 
    }

    CV_lgbm = RandomizedSearchCV(estimator=lgbm_pipeline, param_distributions=lgbm_parameters_model_complexity, scoring = 'r2', cv=5, n_iter=100, verbose=1)
    CV_lgbm.fit(features_train, labels_train)
    lgbm_pipeline.set_params(**CV_lgbm.best_params_)
    lgbm_first_cv_params = CV_lgbm.best_params_
    print(CV_lgbm.best_params_)

    CV_lgbm = RandomizedSearchCV(estimator=lgbm_pipeline, param_distributions=lgbm_parameters_training_speed, scoring = 'r2', cv=5, n_iter=10, verbose=1)
    CV_lgbm.fit(features_train, labels_train)

    print(CV_lgbm.best_params_)

    lgbm_pipeline.set_params(**CV_lgbm.best_params_)
    lgbm_pipeline.fit(features_train, labels_train)

    print(lgbm_pipeline.score(features_test, labels_test))

    joblib.dump(lgbm_pipeline, 'lgbm_pipeline.pkl')

if __name__ == '__main__':

    filename = './apartment_prices.csv'

    if not os.path.isfile(filename):
        red.download_apartment_data(filename=filename)

    df = pd.read_csv(filename, dtype={'postal_code': np.object})

    df = red.preprocess_data(df)

    features_train, features_test, labels_train, labels_test = red.load_train_test(df)

    run_train_test_lgbm()
    pipeline = joblib.load('lgbm_pipeline.pkl')

    known_apts = {'house_type' : ['kt', 'rt', 'rt'],
            'sqm': [42, 75, 115],
            'year_built': [1927, 2019, 1991],
            'elevator': ['on', 'ei', 'ei'],
            'condition': ['hyvä', 'hyvä', 'tyyd.'],
            'plot_ownership':['oma', 'vuokra', 'oma'],
            'postal_code':['00100', '02740', '00740'],
            'floors':[7, 2, 2],
            'current_floor':[3, 2, 2],
            'floor_ratio':[3/7, 1, 1]}



    known_apts_names = ['mathias', 'harry', 'phki']

    known_apartments = pd.DataFrame.from_dict(known_apts)


    sqm_prices = list(pipeline.predict(known_apartments))
    total_prices = list(sqm_prices * known_apartments['sqm'])
    print(f"SQM prices: {sqm_prices}")
    print(f"Total prices: {total_prices}")