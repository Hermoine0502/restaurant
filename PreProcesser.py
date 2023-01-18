import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

class PrePrecesser:
    
    def __init__(self, df_input, info_categorical=None, info_numerical=None):
        self.df = df_input

        if info_categorical is None:
            ls_col_numerical = self.df.select_dtypes(include=np.number).columns.tolist() ### numerical columns
            self.info_categorical = self.df.columns.difference(ls_col_numerical).values.tolist()

        if info_numerical is None:
            self.info_numerical = self.df.columns.difference(self.info_categorical).values.tolist()

    def get_ls_column(self, type):
        if type=="categorical":
            return [x for x in self.df.columns.values.tolist() if x in self.info_categorical]
        elif type=="numerical":
            return [x for x in self.df.columns.values.tolist() if x in self.info_numerical]

    def drop_column_empty(self, empty_proportion_lowerbound=0.2):
        ls_empty_column = self.df.columns[self.df.isnull().mean() >= empty_proportion_lowerbound]
        self.df = self.df.drop(ls_empty_column, axis=1)

    def drop_column_identical(self):
        nunique_ser = self.df.apply(pd.Series.nunique)
        ls_identical_column = nunique_ser[nunique_ser==1].index.values.tolist()
        self.df = self.df.drop(ls_identical_column, axis=1)

    def drop_row_outlier(self):       
        ### BoxPlot Outlier
        Q1 = self.df.quantile(0.25)
        Q3 = self.df.quantile(0.75)
        IQR = Q3 - Q1
        df_isOutlier = ((self.df < (Q1-1.5*IQR))|(self.df > (Q3+1.5*IQR)))
        self.df = self.df[~df_isOutlier.any(axis=1)].reset_index(drop=True)


    def replace_normalize(self, ls_target_column=None):    
        if ls_target_column is None:
            ls_target_column = self.get_ls_column("numerical")   
        self.df[ls_target_column] = StandardScaler().fit_transform(self.df[ls_target_column])


    def onehotencoding(self, ls_target_column=None, ls_head_string=None):
        if ls_target_column is None:
            ls_target_column = self.get_ls_column("categorical")
        self.df = pd.get_dummies(self.df, prefix=ls_head_string, columns=ls_target_column)


    def run(self):
        self.drop_column_empty()
        self.drop_column_identical()
        self.drop_row_outlier()
        self.replace_normalize()
        self.onehotencoding()

    def get_df(self):
        return self.df

if __name__=="__main__":
    file_path = os.path.dirname(os.path.abspath(__file__))
    df_test = pd.read_csv(file_path+"/test_preprocesser.csv")

    PrePrecessTester = PrePrecesser(df_test)
    PrePrecessTester.run()
    df_result = PrePrecessTester.get_df()
    df_result.to_csv(file_path+"/test_preprocesser_result.csv", index=False)

