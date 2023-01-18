"""
@ Created on 2022-01-06

@ author: Frank.Lee

@ purpose: MCF project for data preprocessing
    
@ structure: 
    # libraries
    # user-defined class
        # DataPreprocessing
           ## preprocess_data
        # customError
"""

#region = libraries
import numpy as np
import pandas as pd
#endregion


#region = user-defined-class
class DataPreprocessing():
    """
    Merge input data X, Y and preprocess it.
    """
    def __init__(self, X, Y, target):
        """
        Initialization some parameters.
		
		 Parameters
        -----
        X : dataframe(pandas)
            Process parameters.
        Y : dataframe or series(pandas)
            Characteristic value of each sheet.
        target : str
            the target of characteristic value of each sheet, ex : PSH, CD. 
        """
        self.X = X
        self.Y = Y   
        self.target = target 
        
    def preprocess_data(self):
        """  
        Precrocessing the X, Y data.
        (1) drop Y data first row with 「 PS　特性檢查機」.
        (2) differentiate PSH & CD from Y data and pivot the data to get the dataframe which only has target variables.
        (3) inner join the X,Y data.
        (4) drop features whose missing ratio over 25%.
        (5) impute missing ratio less than 25% with mean.
        (6) impute outliers with 2 standard deviation.
        (7) drop features with unique value.
        (8) raise some CustomError.
          
        Returns
        -----
        df_X : dataframe(pandas)
            Process parameters dataframe after pre-processing.
        df_Y : dataframe(pandas)
            Characteristic value dataframe after pre-processing.            
        """
        
        if self.X.equals(self.Y) and list(self.X.columns)[0] == 'sheet_id':
            raise CustomError(f"X,Y 都放入 X 資料，請重新上傳資料")
        elif self.X.equals(self.Y) and list(self.Y.columns)[0] == 'PS　特性檢查機':
            raise CustomError(f"X,Y 都放入 Y 資料，請重新上傳資料")     
        elif list(self.X.columns)[0] == 'PS　特性檢查機' and list(self.Y.columns)[0] == 'sheet_id':
            raise CustomError(f"X,Y 資料放反，請重新上傳資料")             
            
        # drop Y data first row with 「 PS　特性檢查機」
        correct_header = self.Y.iloc[0]
        Y_new = self.Y[1:] 
        Y_new.columns = correct_header      
        
        # rename 「２次元CODE」&「品名」 for aligning X & Y key column name
        Y_new = Y_new.rename(columns={"２次元CODE":"sheet_id"})
        Y_new = Y_new.rename(columns={"品名":"model_no"})         
        
        # differentiate PSH & CD from Y data
        if self.target == "PSH":
            
            try:
                Y_target = Y_new[(Y_new["缺陷種類"]=="HI") & (Y_new["SUB POINT NO."]==1)]
            except:
                raise CustomError(f"請確認 Y data 是否有「缺陷種類」和「SUB POINT NO.」這兩個欄位")
            
            close_sub_PSH = 0
            
            # there is only one y target
            if len(np.unique(Y_target["判定"].value_counts()))==1:
                close_sub_PSH = 1
            
            try:
                # only get these columns to groupby, it means this sheet_id from same date, model_no, lot_no
                Y_target = Y_target[["sheet_id", "檢查日時", "model_no", "判定", "測定結果"]]
                Y_target["測定結果"] = Y_target["測定結果"].astype(float)    
            except:
                raise CustomError(f"請確認 Y data 是否有「２次元CODE」、「檢查日時」、「品名」、「判定」和「測定結果」欄位")            
            
            Y_target = Y_target.groupby(["sheet_id", "檢查日時","model_no","判定"], as_index=False).mean()
            
            # get the pivot table to get the measurement result
            Y_target_pivot = pd.pivot_table(Y_target, index="sheet_id", columns="判定", values="測定結果")
            Y_target_pivot = Y_target_pivot.reset_index()
            
            # drop missing measurement result
            Y_target_final = Y_target_pivot.dropna()
            Y_target_final = Y_target_final.rename(columns={"L":"Main_PSH"})
            
            if close_sub_PSH == 0:
                Y_target_final = Y_target_final.rename(columns={"O":"Sub_PSH"})    
            
            # calculate the number of y target
            Y_target_num = len(list(Y_target_final.columns))-1
            
        elif self.target == "CD":

            try:
                Y_target = Y_new[(Y_new["缺陷種類"]=="LN") & (Y_new["判定"]=="L")]
            except:
                raise CustomError(f"請確認 Y data 是否有「缺陷種類」和「判定」這兩個欄位")
            
            try:
                # only get these columns to groupby, it means this sheet_id from same date, model_no, lot_no
                Y_target = Y_target[["sheet_id", "檢查日時", "model_no", "SUB POINT NO.", "測定結果"]]
                Y_target["測定結果"] = Y_target["測定結果"].astype(float) 
            except:
                raise CustomError(f"請確認 Y data 是否有「２次元CODE」、「檢查日時」、「品名」、「SUB POINT NO.」和「測定結果」欄位")     
                        
            Y_target = Y_target.groupby(["sheet_id", "檢查日時","model_no","SUB POINT NO."], as_index=False).mean()
            
            # get the pivot table to get the measurement result
            Y_target_pivot = pd.pivot_table(Y_target, index="sheet_id", columns="SUB POINT NO.", values="測定結果")
            Y_target_pivot = Y_target_pivot.reset_index()
            
            # drop missing measurement result
            Y_target_final = Y_target_pivot.dropna()
            Y_target_final = Y_target_final.rename(columns={1:"CDx"})
            Y_target_final = Y_target_final.rename(columns={2:"CDy"}) 
            
            # calculate the number of y target 
            Y_target_num = len(list(Y_target_final.columns))-1
            
        else:
            
            raise CustomError(f"不屬於 PSH or CD, 請再確認分析標的")
        
        # inner join the X,Y data
        df_final = Y_target_final.merge(self.X, on=["sheet_id"], how="inner")
        
        # Replace column names " " with "_"
        df_final.columns = df_final.columns.str.replace(" ", "_")

        if len(df_final) != 0:    
                  
            # drop features whose missing ratio over 25%   
            df_final = df_final.loc[:,df_final.isnull().mean()<.75]
            
            # impute missing ratio less than 25% with mean
            df_final = df_final.fillna(df_final.mean())
                   
            # filter numeric features
            df_final = df_final.select_dtypes(include=["float64","int64"])
    
            # impute outliers with 2 standard deviation
            down_quantiles = df_final.quantile(0.025)
            up_quantiles = df_final.quantile(0.975)      
            df_final[(df_final < down_quantiles)] = np.nan
            df_final = df_final.fillna(down_quantiles)
            df_final[(df_final > up_quantiles)] = np.nan
            df_final = df_final.fillna(up_quantiles)  
                            
            # drop features with unique value
            df_final = df_final[[col for col in list(df_final) if len(df_final[col].unique())>1]]        
                  
            # slice Y sub dataframe
            df_Y = df_final.iloc[:,0:Y_target_num]
            
            # slice X sub dataframe
            df_X = df_final.iloc[:,Y_target_num:]
            
        else:
            
            raise CustomError(f"X, Y 合併後無資料, 請再確認兩資料的 Sheet_id 是否一致")
            
        return df_X, df_Y
#endregion
    
    
#region = user-defined-class
class CustomError(Exception):
    '''
    This is a custom Error type. It's working with 'raise'.
    Usage:
    Only 2 args available or less. Such as:
    1. function name, message (len(args)=2)
    2. only message (len(args)=1)
    3. none
    '''

    def __init__(self, *args):
        if args:
            if len(args) == 2:
                self.fcnName = args[0]
                self.message = args[1]
            elif len(args) == 1:
                self.fcnName = 'CustomError'
                self.message = args[0]
        else:
            self.fcnName = 'CustomError'
            self.message = None

    def __str__(self):
        return '[{0}] {1} '.format(self.fcnName, self.message)
#endregion