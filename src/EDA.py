
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import function_j

train_identity = pd.read_csv('/Users/krc/PycharmProjects/1_fraud_detection/data/train_identity.csv')
train_transaction = pd.read_csv('/Users/krc/PycharmProjects/1_fraud_detection/data/train_transaction.csv')
#print(train_transaction.head())


train_transaction.function_j.cut_column_90()
