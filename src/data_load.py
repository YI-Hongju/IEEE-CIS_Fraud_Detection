#데이터 로드 as dataframe & 간략한 정보
import pandas as pd

def load_data(trans, id, join_df):
#1. trans only
    df = pd.read_csv('./data/csv)
    return
#2. id+trans, inner
    if join_df:
        df_merged_right = pd.merge(train_id_cut,tr_trans_pp, how="inner", on = 'TransactionID')
    return df_merged_right
#3. id_trans, outer