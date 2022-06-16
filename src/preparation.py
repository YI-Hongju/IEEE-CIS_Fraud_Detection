'''
데이터 준비 모듈
    데이터셋 불러오기
    결측치 제거
    인코딩/역인코딩
    스케일링
    컬럼 셀렉션
    다중공선성 계산
    언더샘플링
    주성분분석 및 적용
'''

import pandas as pd

def load_data(trsc, id, is_joined=None,
    is_origin,  
    is_trsc_id_merged,
    train_trsc,
    train_id,
    test_trsc,
    test_id,
    train,
    test
):
#1. Transaction dataset only
    if trans:
        df_trans = pd.read_csv('./0615_train_pp_ver1.csv')
        print(df_trans.describe())
        return df_trans
#2. Use Identity dataset too, but Non-join
    if id:
        df_id = pd.read_csv('./train_identity.csv')
        print(df_id.describe())

        return df_id
#3. Join datasets
    if is_joined == 'outer':
        df_merged_right = pd.merge(df_id, df, how="inner", on = 'TransactionID')
        print(df_merged_right.describe())
        return df_merged_right
    elif is_joined == 'inner':
        df_merged_right = pd.merge(df_id, df, how="outer", on = 'TransactionID')
        print(df_merged_right.describe())
        return df_merged_right
    # elif ...

