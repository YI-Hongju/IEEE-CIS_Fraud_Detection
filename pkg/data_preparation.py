import pandas as pd

def get_df(datasets, is_only=None, join=None):
    #데이터 로드 as dataframe & 간략한 정보

    class DFDatasets():
        def __init__(self, datasets):
            self.sample_submsn = pd.read_csv(datasets.sample_submsn)

    df_datasets = DFDatasets(datasets)

    # Transaction data only
    if is_only == 'transaction':
        df_datasets.train_trsc = pd.read_csv(datasets.train_trsc)
        df_datasets.test_trsc = pd.read_csv(datasets.test_trsc)
        
        print('Load Datasets as DataFrame Succeed!')
        return df_datasets

    # Identity data only
    elif is_only == 'identity':
        df_datasets.train_id = pd.read_csv(datasets.train_id)
        df_datasets.test_id = pd.read_csv(datasets.test_id)

        print('Load Datasets as DataFrame Succeed!')
        return df_datasets

    else: # is_only == None
        if join == 'inner':
            df_datasets.train_trsc = pd.read_csv(datasets.train_trsc)
            df_datasets.train_id = pd.read_csv(datasets.train_id)
            df_datasets.test_trsc = pd.read_csv(datasets.test_trsc)
            df_datasets.test_id = pd.read_csv(datasets.test_id)

            df_datasets.train_merged = datasets.train_trsc.merge(
                datasets.train_id, 
                how=join, 
                on='TransactionID'
            )
            df_datasets.test_merged = datasets.test_trsc.merge(
                datasets.test_id, 
                how=join, 
                on='TransactionID'
            )

            del df_datasets.train_trsc
            del df_datasets.train_id
            del df_datasets.test_trsc
            del df_datasets.test_id

            print(f'Load Datasets as DataFrame and \
Merging as {join} Succeed!')
            return df_datasets
        elif join == 'outer':
            df_datasets.train_trsc = pd.read_csv(datasets.train_trsc)
            df_datasets.train_id = pd.read_csv(datasets.train_id)
            df_datasets.test_trsc = pd.read_csv(datasets.test_trsc)
            df_datasets.test_id = pd.read_csv(datasets.test_id)

            df_datasets.train_merged = datasets.train_trsc.merge(
                datasets.train_id, 
                how=join, # 'outer'
                on='TransactionID'
            )
            df_datasets.test_merged = datasets.test_trsc.merge(
                datasets.test_id, 
                how=join, # 'outer'
                on='TransactionID'
            )

            del df_datasets.train_trsc
            del df_datasets.train_id
            del df_datasets.test_trsc
            del df_datasets.test_id

            print(f'Load Datasets as DataFrame and \
Merging as {join} Succeed!')
            return df_datasets

        else: # join == None
            df_datasets.train_trsc = pd.read_csv(datasets.train_trsc)
            df_datasets.train_id = pd.read_csv(datasets.train_id)
            df_datasets.test_trsc = pd.read_csv(datasets.test_trsc)
            df_datasets.test_id = pd.read_csv(datasets.test_id)

            print(f'Load Datasets as DataFrame Succeed!')
            return df_datasets

def select_col_by_missings(df, threshold): # 결측치에 따라 컬럼 셀력션을 해주는 메소드
    # threshold = 0.1, 0.2, ...
    # 0.2 = 20% "이하"의 결측치를 가진 컬럼만 골라줌

    identity_name = [] # All Columns or 'id' Columns only?

    for i in range(0, len(df.columns)): # Origin code: len(df.iloc[0, :])
        if (df.iloc[:, i].isnull().sum() / len(df.iloc[:, 0])) > threshold: # Origin code: < num
            identity_name.append(df.iloc[:, i].name)
    df = df[identity_name]

def handle_missing_values(df):
    # Handling Missing-data
    # Processing by Column's hierarchy
    # Principles
    #   Numerical: Median
    #   Categorical: Mode

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
    # Email_domains
    mails = ['P_emaildomain', 'R_emaildomain']
    for mail_x in mails:
        if mail_x == 'P_emaildomain':
            # Non missing-values are 84.00%
            # Mode: gmail.com (228355)
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

    # Identity table - Divided into nunique > 100 or not == id_nuniq_{high, lows} [list]
    id_nuniq_lows = [
        'id-01', 'id-03', 'id-04', 'id-05', 'id-07', 'id-08', 
        'id-09', 'id-10', 'id-12', 'id-13', 'id-14', 'id-15', 
        'id-16', 'id-18', 'id-22', 'id-23', 'id-24', 'id-26', 
        'id-27', 'id-28', 'id-29', 'id-30', 'id-32'
    ]
    for id_col in id_nuniq_lows:  # Column is Samll volume in nunique
        if df[id_col].dtype != 'object':  # Numerical, Small unique
            # Numerical인 데이터 : Fill to Median
            df[id_col] = df[id_col].fillna(df[id_col].median())

            # 결측치 50% 이상이면 삭제 추천
            if (df[id_col].isna().sum() / len(df[id_col])) >= 0.5:
                recommends.append(id_col)
        else: # Categorical data
            # NaN이 많은 column 중 'isFraud'가 유의미한 값일 때가 많아,
            # 'Mode'가 아닌 제 3의 카테고리 'Unknown'으로 채움 <- 추가 설명 부탁드립니다!
            df[id_col] = df[id_col].fillna('Unknown')

    id_nuniq_highs = [
        'id-02', 'id-06', 'id-11', 'id-17', 'id-19', 
        'id-20', 'id-21', 'id-25', 'id-31'
    ]
    for id_col in id_nuniq_highs: # Column is Large volume in nunique
        if df[id_col].dtype != 'object': # Numerical and small unique <- ???
            # Numerical인 데이터 : Fill to Median
            df[id_col] = df[id_col].fillna(df[id_col].median())

            # 결측치 50% 이상이면 삭제 추천
            if (df[id_col].isna().sum() / len(df[id_col]) ) >= 0.5:
                recommends.append(id_col)
        else: # Categorical data
            df[id_col] = df[id_col].fillna(df[id_col].mode())

    # Column to Drop
    return recommends

 # 언더샘플링 함수, n = 타겟 컬럼의 n배수의 non 타겟 컬럼 개수를 골라줌
def undersampling(df, n):
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

def main(datasets):
    # Get datasets from data/
    df_datasets = get_df(
        datasets=datasets,
        # is_only='transaction', [예시 코드]
        # join='inner' [예시 코드]
    )
    
    # Column selection
    for df in vars(df_datasets).keys():
        select_col_by_missings(getattr(df_datasets, df), 0.2) # 0.2%

    # Handling missing-values and Recommand lists
    recommand_drop_cols = list()
    for df in vars(df_datasets).keys():
        recommand_drop_cols.append(handle_missing_values(getattr(df_datasets, df)))
    recommand_drop_cols = [col for ls in recommand_drop_cols for col in ls] # 2D array to 1D array

    # Under-sampling
    

    return df_datasets
    