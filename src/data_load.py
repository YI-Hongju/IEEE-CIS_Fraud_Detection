#데이터 로드 as dataframe & 간략한 정보
import pandas as pd

def load(trans, id, join_df):
#1. trans only
    if trans:
        df_trans = pd.read_csv('./0615_train_pp_ver1.csv')
        print(df_trans.describe())
        return df_trans
#2. id

    if id:
        df_id = pd.read_csv('./train_identity.csv')
        print(df_id.describe())

        return df_id
#3. id_trans table with inner join
    if join_df == 'outer ':
        df_merged_right = pd.merge(df_id, df, how="inner", on = 'TransactionID')
        print(df_merged_right.describe())
        return df_merged_right
    if join_df:
        df_merged_right = pd.merge(df_id, df, how="outer", on = 'TransactionID')
        print(df_merged_right.describe())
        return df_merged_right
