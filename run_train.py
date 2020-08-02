import process_data as red
import model_pipelines as mp
import os
import pandas as pd
import numpy as np
import joblib


#ML models and hyperparameter optimisation
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
from lightgbm import LGBMRegressor


def run_train_test_lgbm():
    print("Fitting LGBM model")

    lgbm_pipeline = mp.lgbm_pipeline()

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
    #print(CV_lgbm.best_params_)

    CV_lgbm = RandomizedSearchCV(estimator=lgbm_pipeline, param_distributions=lgbm_parameters_training_speed, scoring = 'r2', cv=5, n_iter=10, verbose=1)
    CV_lgbm.fit(features_train, labels_train)

    #print(CV_lgbm.best_params_)

    lgbm_pipeline.set_params(**CV_lgbm.best_params_)
    lgbm_pipeline.fit(features_train, labels_train)

    score = lgbm_pipeline.score(features_test, labels_test)

    print(score)

    return score, CV_lgbm.best_params_


def run_hyperparameter_optimisation(df, pipeline, parameters, n_iter=100):
    print(f"Running train on {pipeline.named_steps['model']}")
    features_train, features_test, labels_train, labels_test = red.load_train_test(df)


    CV = RandomizedSearchCV(estimator=pipeline, \
                            param_distributions=parameters, \
                            scoring = 'r2', cv=5, n_iter=n_iter, verbose=1)

    CV.fit(features_train, labels_train)

    pipeline.set_params(**CV.best_params_)
    pipeline.fit(features_train, labels_train)

    score = pipeline.score(features_test, labels_test)

    return pipeline, score, CV.best_params_
  


filename = './apartment_prices.csv'

if not os.path.isfile(filename):
    red.download_apartment_data(filename=filename)

df = pd.read_csv(filename)

df = red.preprocess_data(df)

pipelines = mp.lgbm_pipeline(), mp.random_forest_pipeline()
hyperparameters = mp.lgbm_hyperparameters(), mp.random_forest_hyperparameters()
n_iter = [100, 10]

results = {}
for i in range(len(pipelines)):
    model = pipelines[i].named_steps['model']
    pipeline = pipelines[i]
    parameters = hyperparameters[i]
    n = n_iter[i]
    print(f"Run hyperparameter optimisation on {model}")

    pipeline, score, params = run_hyperparameter_optimisation(df,pipeline, parameters, n_iter=n)

    results[model] = {'pipeline': pipeline,
                       'score': score,
                       'parameters': params}



print(results)
    