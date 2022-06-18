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


def main(datasets):
    df_datasets = get_df(
        datasets=datasets,
        # is_only='transaction', [예시 코드]
        # join='inner' [예시 코드]
    )
    
    # TODO: Handling missing-values

    return df_datasets
>>>>>>> a59a4f6 (Print exception statements of processing)
