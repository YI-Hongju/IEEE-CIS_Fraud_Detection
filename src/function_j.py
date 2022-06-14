


class pre_processing:
    # Selecting columns more than --% is filled

    # target columns has to be [1,:],
    # setting [0,:] column as ID
    if __name__ == "__main__":
        pass
    def cut_column_90(self, df):
        identity_name = []
        for i in range(2, len(df.iloc[0,:])  ):
            if (df.iloc[:, i].isnull().sum() / len(df.iloc[:, 0])) < 0.1:
                identity_name.append(df.iloc[:, i].name)
        identity_name.insert(0, df.iloc[:, 0].name)
        df = df[identity_name]
        return df

    def cut_column_80(self, df) :
        identity_name = []
        for i in range(2, len(df.iloc[0,:])  ):
            if (df.iloc[:, i].isnull().sum() / len(df.iloc[:, 0])) < 0.2:
                identity_name.append(df.iloc[:, i].name)
        identity_name.insert(0, df.iloc[:, 0].name)
        df = df[identity_name]
        return df

#다중공선성 계산
    def VIF_cal(self, df):
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        pd.DataFrame({
            "VIF Factor": [variance_inflation_factor(df.values, idx) for idx in range(df.shape[1])],
            "features": df.columns,
        })

    def undersampling(self, df, id_col, n):
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