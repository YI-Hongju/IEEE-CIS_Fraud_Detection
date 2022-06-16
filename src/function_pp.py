from sklearn.preprocessing import LabelEncoder
from collections import defaultdict

class pre_processing:
    # Selecting columns more than --% is filled

#num = 0.1, 0.2 . . .
#0.2 = 20% 이하의 결측치를 가진 컬럼만 골라줌
    if __name__ == "__main__":
        pass

    def cut_column(df, num):
        identity_name = []
        for i in range(0, len(df.iloc[0, :])):
            if (df.iloc[:, i].isnull().sum() / len(df.iloc[:, 0])) < num:
                identity_name.append(df.iloc[:, i].name)
        df = df[identity_name]
        return df

#다중공선성 계산
    def VIF_cal(self, df):
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        pd.DataFrame({
            "VIF Factor": [variance_inflation_factor(df.values, idx) for idx in range(df.shape[1])],
            "features": df.columns,
        })


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