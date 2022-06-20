import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, f1_score, precision_score, roc_auc_score
from xgboost import XGBClassifier, plot_importance
from lightgbm import LGBMClassifier, plot_importance
from catboost import CatBoostClassifier, Pool


# Dataframe memory reduction
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

# Evalutate the model
def get_clf_eval(y_test, y_pred):
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


# Set train, test data
X_train, X_val, y_train, y_val = train_test_split(df_train, y_target, test_size=0.3, random_state=156)


# Running XGB
def XGB():
    xgb_clf = XGBClassifier(
        n_estimators=5000, 
        colsample_bytree=0.75, 
        max_depth=12, 
        min_child_weight=1,
        subsample=0.8,
    #     missing=-999,
        early_stopping_rounds=200
        )

    xgb_clf.fit(X_train, y_train, eval_metric="auc", eval_set=[(X_val, y_val)], verbose = 3)
    xgb_pred = xgb_clf.predict(X_val)
    xgb_pred_proba = xgb_clf.predict_proba(X_val)[:1]

    fig, ax = plt.subplots(figsize=(10,10))
    plot_importance(xgb_clf, ax=ax, max_num_features=50, height=0.4)
    plt.show()

    get_clf_eval(y_val, xgb_pred)

# Running LightGBM
def LGB():
    lgb_clf = LGBMClassifier(
        n_estimators=5000,
        max_depth=20,
        min_data_in_leaf=1000,
        early_stopping_round = 50,
    )

    lgb_clf.fit(X_train, y_train, eval_metric='auc', eval_set=[(X_val, y_val)], verbose = 3)
    lgb_pred = lgb_clf.predict(X_val)
    lgb_pred_proba = lgb_clf.predict_proba(X_val)[:1]

    fig, ax = plt.subplots(figsize=(10,10))
    plot_importance(lgb_clf, ax=ax, max_num_features=50, height=0.4)
    plt.show()

    get_clf_eval(y_val, lgb_pred)

# Running CatBoost
def CAT():
    cat_clf = CatBoostClassifier(
        n_estimators=200,
        depth = 10,
        learning_rate = 0.05, 
        l2_leaf_reg = 30,
        eval_metric='AUC',
        early_stopping_rounds = 100,
        )

    cat_clf.fit(X_train, y_train, eval_set=(X_val, y_val), verbose = 3)
    cat_pred = cat_clf.predict(X_val)
    cat_pred_proba = cat_clf.predict_proba(X_val)[:1]

    get_clf_eval(y_val, cat_pred)

# Main
def main():
    max_scored_model = 
    print('Max AUC score : 'max(roc_auc_score(y_val, xgb_pred),roc_auc_score(y_val, lgb_pred),roc_auc_score(y_val, cat_pred)))
