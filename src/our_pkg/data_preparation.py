import pandas as pd

class DFDatasets():
    def __init__(self, datasets):
        train_trsc = pd.read_csv(datasets.train_trsc)
        train_id = pd.read_csv(datasets.train_id)
        test_trsc = pd.read_csv(datasets.test_trsc)
        test_id = pd.read_csv(datasets.test_id)
        sample_submsn = pd.read_csv(datasets.sample_submsn)

def get_df(datasets):
    
    pass

def main(datasets):
    df_datasets = get_df(datasets)
    
    return df_datasets