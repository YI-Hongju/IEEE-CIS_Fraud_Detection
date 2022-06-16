
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from function_pp import pre_processing

train_identity = pd.read_csv('/Users/krc/PycharmProjects/1_fraud_detection/data/train_identity.csv')
train_transaction = pd.read_csv('/Users/krc/PycharmProjects/1_fraud_detection/data/train_transaction.csv')
#print(train_transaction.head())


pre_processing.cut_column_90(train_transaction)

