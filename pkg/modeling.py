import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns

from xgboost import XGBClassifier, plot_importance
from lightgbm import LGBMClassifier, plot_importance
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, f1_score, precision_score, roc_auc_score
from catboost import CatBoostClassifier, Pool



# Evalutate the model
def get_clf_eval(y_val, y_pred):
    confusion = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    F1 = f1_score(y_test, y_pred)
    AUC = roc_auc_score(y_test, y_pred)
    print('오차행렬:\n', confusion)
    print('\n정확도: {:.4f}'.format(accuracy))
    print('정밀도: {:.4f}'.format(precision))
    print('재현률: {:.4f}'.format(recall))
    print('F1 score: {:.4f}'.format(F1))
    print('AUC score: {:.4f}'.format(AUC))



# Running XGB
def XGBoost(X_train, X_val, y_train, y_val):
    clf = XGBClassifier(
        n_estimators=5000, 
        colsample_bytree=0.75, 
        max_depth=12, 
        min_child_weight=1,
        subsample=0.8,
        # missing=-999,
        early_stopping_rounds=200
    )

    clf.fit(X_train, y_train, eval_metric="auc", eval_set=[(X_val, y_val)], verbose = 3)
    pred = clf.predict(X_val)
    pred_proba = clf.predict_proba(X_val)[:1]

    fig, ax = plt.subplots(figsize=(10,10))
    plot_importance(xgb_clf, ax=ax, max_num_features=50, height=0.4)
    plt.show()

    get_clf_eval(y_val, pred)



# Running LightGBM
def LGBM(X_train, X_val, y_train, y_val):
    lgb_clf = LGBMClassifier(
        n_estimators=5000,
        max_depth=20,
        min_data_in_leaf=1000,
        early_stopping_round = 50,
    )

    lgb_clf.fit(X_train, y_train, eval_metric='auc', eval_set=[(X_val, y_val)], verbose = 3)
    pred = lgb_clf.predict(X_val)
    pred_proba = lgb_clf.predict_proba(X_val)[:1]

    fig, ax = plt.subplots(figsize=(10,10))
    plot_importance(lgb_clf, ax=ax, max_num_features=50, height=0.4)
    plt.show()

    get_clf_eval(y_val, pred)



# Running CatBoost
def CatBoost(X_train, X_val, y_train, y_val):
    cat_clf = CatBoostClassifier(
        n_estimators=200,
        depth = 10,
        learning_rate = 0.05, 
        l2_leaf_reg = 30,
        eval_metric='AUC',
        early_stopping_rounds = 100,
        )

    cat_clf.fit(X_train, y_train, eval_set=(X_val, y_val), verbose = 3)
    pred = cat_clf.predict(X_val)
    pred_proba = cat_clf.predict_proba(X_val)[:1]

    get_clf_eval(y_val, pred)



def main(df_datasets):
    # Memory reducing
    for df in vars(df_datasets).keys():
        reduce_mem_usage(getattr(df_datasets, df))



    # XGBoost(X_train, X_val, y_train, y_val)
    # e.g., XGBoost(df_datasets)
        # X_train = df_datasets.X_train


    
    # XGBoost(X_train, X_val, y_train, y_val)
    # e.g., XGBoost(df_datasets)
        # X_train = df_datasets.X_train

    # XGBoost(X_train, X_val, y_train, y_val)
    # e.g., XGBoost(df_datasets)
        # X_train = df_datasets.X_train



    pass