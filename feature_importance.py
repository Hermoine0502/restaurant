"""
@ Created on 2022-01-06

@ author: Frank.Lee

@ purpose: MCF project for ranking the process parameters X feature importance to facilitate user select controllable process parameters
    
@ structure: 
    # libraries 
    # user-defined class
        # FeatureImportance
           ## ranking_feature  
"""

#region = libraries 
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import pandas as pd
import warnings
#endregion


#region = settings
warnings.filterwarnings("ignore")
#endregion


#region = user-defined class
class FeatureImportance():
    """
    Ranking the process parameters X feature importance to facilitate user select controllable process parameters.
    """    
    def __init__(self, df_X, df_y):
        """
        Initialization some parameters.
		
		 Parameters
        -----
        df_X : dataframe(pandas)
            Process parameters dataframe after pre-processing.
        df_y : dataframe(pandas) or series(pandas)
            Characteristic value dataframe after pre-processing.   
        """
        self.df_X = df_X
        self.df_y = df_y   
        
    def ranking_feature(self):
        """
        Ranking the process parameters X feature importance.
        (1) feature scaling.
        (2) get the feature importances ranking by random forest in terms of multiple target.
              
        Returns
        -----
        df_feature_rank : dataframe(pandas)
            features and its importance by descending order.
        """  
	         
        # standard scaling       
        X_features_list = self.df_X.columns.tolist()      
        std_sclr = StandardScaler()
        std_sclr.fit(self.df_X)
        df_std_sclr_X = std_sclr.transform(self.df_X)      
        df_std_sclr_X = pd.DataFrame(df_std_sclr_X, columns = X_features_list)
        
        # create multi target regression model 
        regr_rf = RandomForestRegressor(n_estimators=1024, random_state=42) 
        regr_rf = regr_rf.fit(df_std_sclr_X, self.df_y)
        
        # get the feature importances 
        df_feature_rank = pd.DataFrame(
            {"feature": list(self.df_X.columns),
             "importance": list(regr_rf.feature_importances_)
            }).sort_values(by="importance", ascending=False) 
        
        # transform unit to percentile (total equal to 100)
        df_feature_rank["importance"] = 100 * df_feature_rank["importance"]
            
        return df_feature_rank
#endregion