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
        def __init__(self, train_trsc, train_id, test_trsc, test_id, sample_submsn):
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
        sample_submsn=sample_submsn
    )

    datasets = data_preparation.main(datasets)

    processed_submsn = modeling.main(datasets)
    return processed_submsn

processed_submsn = get_submission(
    train_trsc='train_transaction.csv',
    train_id='train_identity.csv',
    test_trsc='test_transaction.csv',
    test_id='test_identity.csv',
    sample_submsn='sample_submission.csv',
    status='raw' # raw / processed / pended
)
processed_submsn.to_csv('./out/submission.csv')





















# 아래 코드는 신경쓰지 마세요!

# class Datasets():
#     def __init__(self, is_origin_data=True, train=None, test=None):
#         if is_origin_data:
#             self.train_trsc = pd.read_csv('./data/train_transaction.csv')
#             self.train_id = pd.read_csv('./data/train_identity.csv')
#             self.test_trsc = pd.read_csv('./data/test_transaction.csv')
#             self.test_id = pd.read_csv('./data/test_identity.csv')

#             self.train = self.train_trsc.merge(
#                 self.train_id,
#                 how='left',
#                 on='TransactionID'
#             )
#             self.test = self.test_trsc.merge(
#                 self.test_id,
#                 how='left',
#                 on='TransactionID'
#             )
#             del self.train_trsc
#             del self.train_id
#             del self.test_trsc
#             del self.test_id

#             self.sample_sbmision = pd.read_csv('./data/sample_submission.csv')
#         else:
#             self.base_path = './data/'

#             self.train = pd.read_csv(self.base_path + train)
#             self.test = pd.read_csv(self.base_path + test)
#             self.sample_sbmision = pd.read_csv('./data/sample_submission.csv')

# datasets_00 = Datasets(
#     is_origin_data=False,
#     train='ver01_0615.csv', 
#     test='ver01_0615_test.csv'
# )
