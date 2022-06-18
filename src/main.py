import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
from pandas_profiling import ProfileReport


# train_id = pd.read_csv("/Volumes/Samsung USB/yeardream/modeling_miniPRJ/data/train_identity.csv")
train_trsc = pd.read_csv("/Volumes/Samsung USB/yeardream/modeling_miniPRJ/data/train_transaction.csv")

# df_train_id = pd.DataFrame(train_id)
df_train_trsc = pd.DataFrame(train_trsc)

# profile1 = ProfileReport(df_train_id)
profile2 = ProfileReport(df_train_trsc, 
correlations={"cramers": {"calculate": False},"pearson": {"calculate": False},"spearman": {"calculate": False},"kendall": {"calculate": False},"phi_k": {"calculate": False}})

# profile1.to_file("report_train_id.html")
profile2.to_file("report_train_trsc.html")