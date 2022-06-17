<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> 42b13c2605aab02601eaa17fc745c20d1289a796
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

<<<<<<< HEAD
=======
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib as plt
import numpy as np

class col_processing:
    # Selecting columns more than --% is filled

#num = 0.1, 0.2 . . .
#0.2 = 20% 이하의 결측치를 가진 컬럼만 골라줌
    def cut_column(df, num):
        identity_name = []
        for i in range(0, len(df.iloc[0, :])):
            if (df.iloc[:, i].isnull().sum() / len(df.iloc[:, 0])) < num:
                identity_name.append(df.iloc[:, i].name)
        df = df[identity_name]
        return df


    def apply_PCA(X, n_components, show_plot):
        from sklearn.decomposition import PCA
        pca = PCA(n_components=n_components)  # if 0.9 = 원래 데이터의 90%를 보존
        pca_090 = pca.fit(X)  # 학습 및 변환
        reduced_X = pca_090.transform(X)
        print("explained variance ratio:", pca.explained_variance_ratio_) #분산 비율
        print("shape:", reduced_X.shape) #PCA df shape

        if show_plot:
            labels = [f"PC{x}" for x in range(1, reduced_X.shape[1] + 1)]
            pca_090_variance = np.round(pca_090.explained_variance_ratio_.cumsum() * 100, decimals=1)
            plt.figure(figsize=(25, 5))
            plt.bar(x=range(1, len(pca_090_variance) + 1), height=pca_090_variance, tick_label=labels)

            plt.xticks(rotation=90, color='indigo', size=15)
            plt.yticks(rotation=0, color='indigo', size=15)
            plt.title('Scree Plot', color='tab:orange', fontsize=25)
            plt.xlabel('Principal Components', {'color': 'tab:orange', 'fontsize': 15})
            plt.ylabel('Cumulative percentage of explained variance ', {'color': 'tab:orange', 'fontsize': 15})
            plt.show()

        pca_df = pd.DataFrame(reduced_X, columns=labels)
        print(pca_df)

        return pca_df

        # 다중공선성 계산
    def VIF_cal(self, df):
        VIF_table = pd.DataFrame({
            "VIF Factor": [variance_inflation_factor(df.values, idx) for idx in range(df.shape[1])],
            "features": df.columns,
        })
        return VIF_table


    #train&test column 동시에 column drop
    def train_test_processing(self, df_tr, df_te, col_list):
        total = pd.concat([df_tr, df_te])
        split_point = len(df_tr)

        # drop
        tempX = total.drop(columns=col_list, axis =1)
        df_train = tempX[:split_point]
        df_test = tempX[split_point:]
        print(df_train.shape, df_test.shape)
        return df_train, df_test


#범주형 column 인코딩, copied from s park
class MultiColLabelEncoder:
    def __init__(self):
        self.encoder_dict = defaultdict(LabelEncoder)

    def fit_transform(self, X: pd.DataFrame, columns: list):  # 컬럼명 리스트 기준으로 레이블인코딩
        if not isinstance(columns, list):
            columns = [columns]

        output = X.copy()
        output[columns] = X[columns].apply(lambda x: self.encoder_dict[x.name].fit_transform(x))

        return output

    def inverse_transform(self, X: pd.DataFrame, columns: list):  # 인코딩 된 열 레이블 복구
        if not isinstance(columns, list):
            columns = [columns]

        if not all(key in self.encoder_dict for key in columns):
            raise KeyError(f'At least one of {columns} is not encoded before')

        output = X.copy()
        try:
            output[columns] = X[columns].apply(lambda x: self.encoder_dict[x.name].inverse_transform(x))
        except ValueError:
            print(f'Need assignment when do "fit_transform" function')
            raise

        return output

class row_processing:
    # 언더샘플링 함수, n = 타겟 컬럼의 n배수의 non 타겟 컬럼 개수를 골라줌줌
    def undersampling(self, df, n):
        # Find Number of samples which are Fraud
        no_frauds = len(df[df['id_col'] == 1]) * n  # 열배!
        # Get indices of non fraud samples
        non_fraud_indices = df[df.id_col == 0].index
        # Random sample non fraud indices
        random_indices = np.random.choice(non_fraud_indices, no_frauds, replace=False)
        # Find the indices of fraud samples
        fraud_indices = df[df.isFraud == 1].index
        # Concat fraud indices with sample non-fraud ones
        under_sample_indices = np.concatenate([fraud_indices, random_indices])
        # Get Balance Dataframe
        under_sample = df.loc[under_sample_indices]
        return under_sample

    def main(df):
        # Handling Missing-data
        # Processing by Column's hierarchy
        # Numerical: Median
        # Categorical: Mode

        recommends = list()  # Recommend features to drop

        # card2 - 6
        # All missing-values are less than 1.6%
        n_cards = ['card2', 'card3', 'card5']
        for card_x in n_cards:
            df[card_x] = df[card_x].fillna(df[card_x].median())
        # All missing-values are less than 0.26%
        c_cards = ['card4', 'card6']
        for card_x in c_cards:
            if card_x == 'card4':
                # visa: 65.32%
                # mastercard: 32.12%
                # american express: 1.41%
                # discover: 1.12%
                # To 'visa'
                df[card_x] = df[card_x].fillna('visa')
            else:  # elif card_x == 'card6'
                # debit: 74.69%
                # credit: 25.29%
                # debit or credit: 0.005%
                # charge card: 0.002%
                df[card_x] = df[card_x].fillna('debit')

        # addr1 - 2
        addrs = ['addr1', 'addr2']
        for addr_x in addrs:
            if addr_x == 'addr1':
                df[addr_x] = df[addr_x].fillna(df[addr_x].median())
            else:  # elif addr_x == 'addr2'
                df[addr_x] = df[addr_x].fillna(df[addr_x].median())

        # dist1 - 2
        dists = ['dist1', 'dist2']
        for dist_x in dists:
            if dist_x == 'dist1':
                # Non missing-values are 40.34%
                # Median: 118.50
                # Mean: 8.0 -> Selected
                # Range: 0 ~ 10286
                df[dist_x] = df[dist_x].fillna(df[dist_x].median())
            else:  # elif dist_x == 'dist2':
                # Non-null values are 37627
                # Non missing-values are 93.62%

                # Range: 0.0 ~ 11623.0
                # Median:  37.0
                # Mean:  231.85

                # train_merged.dist2[
                #     (train_merged.dist2 >= 230) & (train_merged.dist2 <= 232)
                # ].value_counts()
                # 232.0    22
                # 230.0     7
                # 231.0     3
                # Name: dist2

                # train_merged.dist2.value_counts()
                # 7.0       5687 -> Selected
                # 0.0       3519
                # 1.0       1374
                # 9.0        742
                # 4.0        659
                df[dist_x] = df[dist_x].fillna(df[dist_x].mode()[0])
                recommends.append(dist_x)
        #email_domains
        mails = ['P_emaildomain', 'R_emaildomain']
        for mail_x in mails:
            if mail_x == 'P_emaildomain':
                # Non missing-values are 84.00%
                # Mode:gmail.com (228355)
                # N of unique: 59
                df[mail_x] = df[mail_x].fillna('gmail.com')
            else:  # elif mail_x == 'R_emaildomain':
                # Null values are 453249
                # Non missing-values are 23.24%
                # Mode:gmail.com (57147)
                # N of unique: 60

                # train_merged.dist2.value_counts()
                # 7.0       5687 -> Selected
                # 0.0       3519
                # 1.0       1374
                # 9.0        742
                # 4.0        659
                df[mail_x] = df[mail_x].fillna('gmail.com')
                recommends.append(mail_x)

        idnu_over100 = ['id_02', 'id_06', 'id_11', 'id_17', 'id_19', 'id_20', 'id_21', 'id_25', 'id_31']
        for id_2 in idnu_over100: #large nunique
            if df[id_2].dtype != 'object'  : #numerical, small unique
                #Numerical 인 데이터 : median으로 채움
                df[id_2] = df[id_2].fillna(df[id_2].median())
                # 결측치 50% 이상이면 삭제 추천
                if (df[id_2].isna().sum() / len(df[id_2]) ) >0.5 :
                        recommends.append(id_2)
            else:  # categorical
                df[id_2] = df[id_2].fillna(df[id_2].mode())


            # identity table - divided into nunique > 100 or not
            idnu_under100 = ['id_01', 'id_03', 'id_04', 'id_05', 'id_07', 'id_08', 'id_09', 'id_10', 'id_12', 'id_13', 'id_14',
                             'id_15', 'id_16', 'id_18', 'id_22', 'id_23', 'id_24', 'id_26', 'id_27', 'id_28', 'id_29', 'id_30',
                             'id_32']
        for id_1 in idnu_under100:  # smallnunique
            if df[id_1].dtype != 'object':  # numerical, small unique
                # Numerical 인 데이터 : median으로 채움
                df[id_1] = df[id_1].fillna(df[id_1].median())
                # 결측치 50% 이상이면 삭제 추천
                if (df[id_1].isna().sum() / len(df[id_1])) > 0.5:
                    recommends.append(id_1)
        else:  # categorical
            # Nan이 많은 column 중 isFraud 가 유의미한 값일 때가 많아 mode가 아닌 제3의 카테고리로 채움
            df[id_1] = df[id_1].fillna('Unknown')

        return df, recommends
>>>>>>> b_0617
=======
>>>>>>> 42b13c2605aab02601eaa17fc745c20d1289a796
