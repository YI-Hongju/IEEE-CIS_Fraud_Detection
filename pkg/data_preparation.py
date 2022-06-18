import pandas as pd

def get_df(datasets, is_only=None, join=None):
    #데이터 로드 as dataframe & 간략한 정보

    class DFDatasets():
        def __init__(self, datasets):
            self.train_trsc = pd.read_csv(datasets.train_trsc)
            self.train_id = pd.read_csv(datasets.train_id)
            self.test_trsc = pd.read_csv(datasets.test_trsc)
            self.test_id = pd.read_csv(datasets.test_id)
            self.sample_submsn = pd.read_csv(datasets.sample_submsn)

    df_datasets = DFDatasets(datasets)

    # Transaction data only
    if is_only == 'transaction':
        del df_datasets.train_id
        del df_datasets.test_id

        print('Load Datasets as DataFrame Succeed!')
        return df_datasets

    # Identity data only
    elif is_only == 'identity':
        del df_datasets.train_trsc
        del df_datasets.test_trsc

        print('Load Datasets as DataFrame Succeed!')
        return df_datasets

    else: # is_only == None
        if join == 'inner':
            df_datasets.train_merged = datasets.train_trsc.merge(
                datasets.train_id, 
                how="inner", 
                on='TransactionID'
            )
            df_datasets.test_merged = datasets.test_trsc.merge(
                datasets.test_id, 
                how="inner", 
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
            df_datasets.train_merged = datasets.train_trsc.merge(
                datasets.train_id, 
                how="outer", # 'outer'
                on='TransactionID'
            )
            df_datasets.test_merged = datasets.test_trsc.merge(
                datasets.test_id, 
                how="outer", # 'outer'
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
            print(f'Load Datasets as DataFrame Succeed!')

            return df_datasets

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

def select_col_by_missings(df, threshold):
    # threshold = 0.1, 0.2, ...
    # 0.2 = 20% 이하의 결측치를 가진 컬럼만 골라줌

    identity_name = []

    for i in range(0, len(df.columns)):
        if (df.iloc[:, i].isnull().sum() / len(df.iloc[:, 0])) < num:
            identity_name.append(df.iloc[:, i].name)
    df = df[identity_name]
    return df

def main(datasets):
    df_datasets = get_df(
        datasets=datasets,
        # is_only='transaction', [예시 코드]
        # join='inner' [예시 코드]
    )
    
    # Handling missing-values
    handle_missing_values(df_datasets)

    # Column selection
    select_col_by_missings()

    # Under-sampling

    return df_datasets