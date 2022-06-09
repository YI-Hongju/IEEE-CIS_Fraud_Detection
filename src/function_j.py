

#Selecting columns more than --% is filled

#target columns has to be [1,:],
#setting [0,:] column as ID
class pre_processing:
    if __name__ == "__main__":
        pass
    def cut_column_90(df):
        identity_name = []
        for i in range(2, len(df.iloc[0,:])  ):
            if (df.iloc[:, i].isnull().sum() / len(df.iloc[:, 0])) < 0.1:
                identity_name.append(df.iloc[:, i].name)
        identity_name.insert(0, df.iloc[:, 0].name)
        df = df[identity_name]
        return df
        print(df.info())

    def cut_column_50(df) :
        identity_name = []
        for i in range(2, len(df.iloc[0,:])  ):
            if (df.iloc[:, i].isnull().sum() / len(df.iloc[:, 0])) < 0.5:
                identity_name.append(df.iloc[:, i].name)
        identity_name.insert(0, df.iloc[:, 0].name)
        df = df[identity_name]
        return df
        print(df.info())