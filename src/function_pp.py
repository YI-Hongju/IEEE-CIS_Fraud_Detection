import pandas as pd
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib as plt
import numpy as np

class pre_processing:
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

#다중공선성 계산
    def VIF_cal(self, df):
        VIF_table = pd.DataFrame({
            "VIF Factor": [variance_inflation_factor(df.values, idx) for idx in range(df.shape[1])],
            "features": df.columns,
        })
        return VIF_table


#언더샘플링 함수, n = 타겟 컬럼의 n배수의 non 타겟 컬럼 개수를 골라줌줌
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

    def apply_PCA(X, n_components, show_plot):
        from sklearn.decomposition import PCA
        # training data와 test data를 모두 PCA를 이용하여 차원 감소를 수행합니다.
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
        display(pca_df)

        return pca_df
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