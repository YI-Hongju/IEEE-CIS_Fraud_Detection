<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> 42b13c2605aab02601eaa17fc745c20d1289a796
import preparation as pp
import modeling as mdling

import pandas as pd

def get_submission(
    is_train_test_only,
    train_trsc,
    train_id,
    test_trsc,
    test_id,
    sample_submsn,
    train_test
):
    if is_train_test_only:
        pass
    else:
        pass

    return my_submission

my_submission = get_submission(
    is_train_test_only=True,
    train_trsc=None,
    train_id=None,
    test_trsc=None,
    test_id=None,
    sample_submsn='sample_submission.csv'
    train_test=['ver_01_0615.csv', 'ver_01_0615_test.csv']
)
my_submission.to_csv('./out/submission.csv')


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
<<<<<<< HEAD
# )
=======
from data_load import load
from preparation import pre_processing
# import function_pp
#import numpy as np
#import seaborn as sns
#import matplotlib.pyplot as plt
#from function_pp import pre_processing

#load_trans = data_load.load(True, False)
#print(load_trans)


x = load(True, True, join_df = 'inner')
print(x.info())


>>>>>>> b_0617
=======
# )
>>>>>>> 42b13c2605aab02601eaa17fc745c20d1289a796
