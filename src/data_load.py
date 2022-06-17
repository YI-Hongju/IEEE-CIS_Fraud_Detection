#데이터 로드 as dataframe & 간략한 정보
import pandas as pd

def load(trans, id, join_df):
    if trans & id:
        if join_df:
            df_trans = pd.read_csv('./0615_train_pp_ver1.csv')
            df_id = pd.read_csv('./train_identity.csv')
            df_merged = pd.merge(df_id, df_trans, how="outer", on='TransactionID')
            print(df_merged.describe())
            return df_merged
        else:
            df_merged = pd.merge(df_id, df_trans, how="inner", on='TransactionID')
            print(df_merged.describe())
            return df_merged

    # trans only
    if trans:
        df_trans = pd.read_csv('./0615_train_pp_ver1.csv')
        print(df_trans.describe())
        return df_trans
    # id only
    if id:
        df_id = pd.read_csv('./train_identity.csv')
        print(df_id.describe())

        return df_id
#3. id_trans table, default = outer join
