from our_pkg import data_preparation
from our_pkg import modeling

import pandas as pd

def get_submission(
    train_trsc,
    train_id,
    test_trsc,
    test_id,
    sample_submsn,
    status
):    
    class Datasets():
        def __init__(self, train_trsc, train_id, test_trsc, test_id, sample_submsn, status):
            if status == 'raw':
                self.base_path = './data/raw/'
            elif status == 'processed':
                self.base_path = './data/processed/'
            elif status == 'pended':
                self.base_path = './data/pended/'
            else: 
                print('[\'status\'] value error.')
                return False

            self.train_trsc = self.base_path + train_trsc
            self.train_id = self.base_path + train_id
            self.test_trsc = self.base_path + test_trsc
            self.test_id = self.base_path + test_id
            self.sample_submsn = self.base_path + sample_submsn

        # 요건 신경쓰지 마셔유들~
        # def print_id(self):
        #     datas = [train_trsc, train_id, test_trsc, test_id, sample_submsn]

        #     print('Location on Memory:')
        #     for data in datas:
        #         print(id(self.data))

    datasets = Datasets(
        train_trsc=train_trsc,
        train_id=train_id,
        test_trsc=test_trsc,
        test_id=test_id,
        sample_submsn=sample_submsn,
        status=status
    )

    df_datasets = data_preparation.main(datasets)
    if df_datasets:
        print('Data preparation Succeed!\nInit Modeling session...')
    else:
        print('Data preparation Failed!\nCannot Init Modeling session...')
        return False

    # processed_submsn = modeling.main(df_datasets) [코드 시험 운행을 위해 주석 처리한 것, 실제로 사용될 코드!]
    # if processed_submsn:
    #     print('Main processing Succeed!\nMake .csv file...')
    # else:
    #     print('Main processing Failed!\nCannot Make .csv file...')
    #     return False
    # return processed_submsn

processed_submsn = get_submission(
    train_trsc='train_transaction.csv',
    train_id='train_identity.csv',
    test_trsc='test_transaction.csv',
    test_id='test_identity.csv',
    sample_submsn='sample_submission.csv',
    status='raw' # raw / processed / pended
)
# processed_submsn.to_csv('./out/submission.csv') [코드 시험 운행을 위해 주석 처리한 것, 실제로 사용될 코드!]
# if processed_submns:
#     print('All processes are Succeed!\nCheck ./out/submission.csv!')
# else:
#     print('Makint .csv file Failed!')
#     return False
<<<<<<< HEAD
>>>>>>> 04960a0 (Build data_praparation.get_df())
=======
>>>>>>> a59a4f6 (Print exception statements of processing)
