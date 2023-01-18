"""
@ Created on 2022-01-06

@ author: Frank.Lee

@ purpose: MCF project for automatically generating the best model and evaluated by minimum MAPE
    
@ structure: 
    # libraries 
    # user-defined class
        # ModelGenerator
           ## mean_absolute_percentage_error 
           ## select_best_model    
           ## tune_hyperparameter
           ## run 
"""

#region = libraries 
import pandas as pd
import numpy as np
from numpy import mean
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import make_scorer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Lasso
from sklearn.svm import SVR
from xgboost import XGBRegressor
import lightgbm as lgb
from hyperopt import tpe, hp, fmin, space_eval
from hyperopt.early_stop import no_progress_loss
import datetime
import warnings
#endregion


#region = settings
warnings.filterwarnings("ignore")
seed = 42
n_iter = 50
#endregion


#region = user-defined class
class ModelGenerator():
    """
    Generate best model from preprocessing data and evaluated by minimum MAPE.
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
        
    def mean_absolute_percentage_error(self, y_true, y_predict,**kwargs):
        """
        The mean absolute percentage error.
        
    	 Parameters
        -----
        y_true : series(pandas)
            the actual data value.
        y_predict : series(pandas)
            the predicted data value.
            
        Returns
        -----
        score : float64
            the value of mean absolute percentage error.
        """          
        y_true = np.array(y_true)
        y_predict = np.array(y_predict)
        score = np.mean(np.abs((y_true - y_predict) / y_true)) * 100
         
        return score
               
    def select_best_model(self, df_std_sclr_X):
        """
        Select best model from preprocessing data and evaluated by minimum MAPE among 6 models.
        
    	 Parameters
        -----
        df_std_sclr_X : dataframe(pandas)
            the X dataframe after standard scaling.
            
        Returns
        -----
        best_model_name : str
            best model name which standed out among all models and evaluated by cross validation minimum MAPE.       
        best_model : sklearn model(multi-output)
            best model which standed out among all models which evaluated by cross validation with minimum MAPE; moreover, the model can predict multiple output.
        best_MAPE : float
            the MAPE of best model.
        """  
        
        # dictionary of all available models
        dict_models = {        
            "KNN" : KNeighborsRegressor(),
            "Random Forest" : RandomForestRegressor(random_state=seed),
            "Lasso Regression" : MultiOutputRegressor(Lasso(random_state=seed)),
            "SVR" : MultiOutputRegressor(SVR()),            
            "XGBoost" : MultiOutputRegressor(XGBRegressor(random_state=seed)),            
            "LightGBM" : MultiOutputRegressor(lgb.LGBMRegressor(random_state=seed))
        }   
             
        #initiate best MAPE value to the max 100                 
        best_MAPE = 100
        
        # define Repeated KFold
        cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=seed)
         
        # fit all model and get the mimimum MAPE of all available models         
        for model_name, model_temp in dict_models.items():      

            stage1_start_time = datetime.datetime.now()
            
            # evaluate the model and collect the MAPE scores
            temp_MAPE = cross_val_score(model_temp, df_std_sclr_X, self.df_y, scoring=make_scorer(self.mean_absolute_percentage_error, greater_is_better=True), cv=cv)
            # get the k-fold average MAPE value
            temp_MAPE = mean(temp_MAPE)
            
            # calculate the model training time
            stage1_end_time = datetime.datetime.now()
            stage1_runnung_time = stage1_end_time - stage1_start_time
            stage1_runnung_time = float(stage1_runnung_time.total_seconds())
            
            # summarize information
            print(f"{model_name} \nMAPE: {temp_MAPE} \nRunTime: {stage1_runnung_time}\n")
            
            # the algorithm to choose best model
            if temp_MAPE < best_MAPE:
                best_MAPE = temp_MAPE
                best_model = model_temp
                best_model_name = model_name
        
        return best_model_name, best_model, best_MAPE

    def hyperparameter_tuning(self, df_std_sclr_X, best_model_name, best_model, best_MAPE):
        """
        Bayesian optimization to find the best hyperparameters from the best model.
        
    	 Parameters
        -----
        df_std_sclr_X : dataframe(pandas)
            the X dataframe after standard scaling.
        best_model_name : str
            best model name which standed out among all models and evaluated by cross validation minimum MAPE.       
        best_model : sklearn model(multi-output)
            best model which standed out among all models which evaluated by cross validation with minimum MAPE; moreover, the model can predict multiple output.
        best_MAPE : float
            the MAPE of best model.
            
        Returns
        -----     
        best_bo_model : sklearn model(multi-output)
            best model which standed out among all models which evaluated by cross validation minimum MAPE and use bayesian optimization to find the best hyperparameters; moreover, the model can predict multiple output.
        best_bo_MAPE : float
            the MAPE of best model after bayesian optimization.
        """ 

        # define the candidate models' parameters for hyperparameters tuning
        knn_parameters = {
            'n_neighbors'   : hp.choice('n_neighbors',np.arange(2,30,1,dtype=int))
            }
        
        rf_parameters = {
            'n_estimators'  : hp.choice('n_estimators',np.arange(100,1000,5,dtype=int)),
            'max_depth'     : hp.choice('max_depth',np.arange(2,16,1,dtype=int))
            }  
        
        lasso_parameters = {
            'alpha'         : hp.choice('alpha',np.arange(0,5,0.002,dtype=float))
            }  
        
        svr_parameters = {
            'C'             : hp.quniform('C',0.1,50,0.1),
            'kernel'        : hp.choice('kernel',['linear','poly','rbf','sigmoid']),
            'gamma'         : hp.quniform('gamma',0.001,1,0.001)          
            }    
        
        xgb_parameters = {
            'max_depth'     : hp.choice('max_depth',np.arange(2,16,1,dtype=int)),
            'n_estimators'  : hp.choice('n_estimators',np.arange(100,1000,10,dtype=int)),
            'learning_rate' : hp.choice('learning_rate',np.arange(0.001,1.0,0.001,dtype=float))   
            }

        lgb_parameters = {
            'max_depth'     : hp.choice('max_depth',np.arange(2,16,1,dtype=int)),
            'n_estimators'  : hp.choice('n_estimators',np.arange(100,1000,10,dtype=int))
            }        

        if best_model_name == "KNN":
            parameters = knn_parameters
        elif best_model_name == "Random Forest":
            parameters = rf_parameters
        elif best_model_name == "Lasso Regression":
            parameters = lasso_parameters
        elif best_model_name == "SVR":
            parameters = svr_parameters
        elif best_model_name == "XGBoost":
            parameters = xgb_parameters
        elif best_model_name == "LightGBM":
            parameters = lgb_parameters
            
        stage2_start_time = datetime.datetime.now()   

        # define Repeated KFold
        cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=seed)

        # define the loss function
        def loss(params):

            if best_model_name == "KNN":
                rgsr = KNeighborsRegressor(**params)
            elif best_model_name == "Random Forest":
                rgsr = RandomForestRegressor(**params,random_state=seed)
            elif best_model_name == "Lasso Regression":
                rgsr = MultiOutputRegressor(Lasso(**params,random_state=seed))
            elif best_model_name == "SVR":
                rgsr = MultiOutputRegressor(SVR(**params)) 
            elif best_model_name == "XGBoost":
                rgsr = MultiOutputRegressor(XGBRegressor(**params,random_state=seed))
            elif best_model_name == "LightGBM":
                rgsr = MultiOutputRegressor(lgb.LGBMRegressor(**params,random_state=seed))
                            
            score = cross_val_score(rgsr, df_std_sclr_X, self.df_y, scoring="neg_mean_absolute_error").mean()
        
            return -score     
        
        # find the best bayesian optimization parameters
        best_bo_parameter = fmin(
                            fn=loss,                                            # loss function
                            space=parameters,                                   # parameter spaces
                            algo=tpe.suggest,                                   # algorithm
                            max_evals=n_iter,                                   # iteration times
                            rstate=np.random.RandomState(seed),                 # random seed
                            early_stop_fn=no_progress_loss(percent_increase=5)  # early stop
                            )    
        
        # print out the best bayesian optimization parameters
        best_bo_parameter = space_eval(parameters, best_bo_parameter)
        print(f"best parameter : {best_bo_parameter}")
        
        # generate the best model
        if best_model_name == "KNN":
            best_bo_model = KNeighborsRegressor(**best_bo_parameter)
        elif best_model_name == "Random Forest":
            best_bo_model = RandomForestRegressor(**best_bo_parameter,random_state=seed)
        elif best_model_name == "Lasso Regression":
            best_bo_model = MultiOutputRegressor(Lasso(**best_bo_parameter,random_state=seed))
        elif best_model_name == "SVR":
            best_bo_model = MultiOutputRegressor(SVR(**best_bo_parameter)) 
        elif best_model_name == "XGBoost":
            best_bo_model = MultiOutputRegressor(XGBRegressor(**best_bo_parameter,random_state=seed))
        elif best_model_name == "LightGBM":
            best_bo_model = MultiOutputRegressor(lgb.LGBMRegressor(**best_bo_parameter,random_state=seed))
        
        # evaluate the best model and collect the MAPE scores
        best_bo_MAPE = cross_val_score(best_bo_model, df_std_sclr_X, self.df_y, scoring=make_scorer(self.mean_absolute_percentage_error, greater_is_better=True), cv=cv).mean()
        
        # sometimes cross validation will come out a smaller value in MAPE
        if best_MAPE < best_bo_MAPE:
            best_bo_MAPE = best_MAPE
            best_bo_model = best_model.fit(df_std_sclr_X, self.df_y)
        else:
            # fit the best model generated from bayesian optimization
            best_bo_model = best_bo_model.fit(df_std_sclr_X, self.df_y)   
       
        # calculate the bayesian optimization searching time
        stage2_end_time = datetime.datetime.now()
        stage2_runnung_time = stage2_end_time - stage2_start_time
        stage2_runnung_time = float(stage2_runnung_time.total_seconds())

        # summarize information
        print(f"Best MAPE after bayesian optimization : {best_bo_MAPE} \nTraining Time : {stage2_runnung_time}\n")         
        
        return best_bo_model, best_bo_MAPE 
    
    def run(self):
        """
        Generate best model from preprocessing data and evaluated by minimum MAPE.
        
        * preprocessing X, y
          - feature scaling.
        * stage1 -> select_best_model
          - choose best model from 6 models  
          - cv RepeatedKFold(n_splits=10, n_repeats=3)
        * Stage 2 -> hyperparameter_tuning
          - bayesian optimization to search the hyperparameters from best model

        Returns
        -----
        best_model_name : str
            best model name which standed out among all models and evaluated by cross validation minimum MAPE.       
        best_bo_model : sklearn model(multi-output)
            best model which standed out among all models which evaluated by cross validation minimum MAPE and use bayesian optimization to find the best hyperparameters; moreover, the model can predict multiple output.
        best_bo_MAPE : float
            the MAPE of best model after bayesian optimization.
        std_sclr : scaling model in fitting X data
            for parameter_recommend module use.
        """  
	         
        # standard scaling       
        X_features_list = self.df_X.columns.tolist()      
        std_sclr = StandardScaler()
        std_sclr.fit(self.df_X)
        df_std_sclr_X = std_sclr.transform(self.df_X)      
        df_std_sclr_X = pd.DataFrame(df_std_sclr_X, columns = X_features_list) 
        
        # call select_best_model method to get the best_model_name, best_model, best_MAPE
        best_model_name, best_model, best_MAPE = self.select_best_model(df_std_sclr_X)
        
        # call hyperparameter_tuning method to get the best_model_name, best_model, best_MAPE
        best_bo_model, best_bo_MAPE = self.hyperparameter_tuning(df_std_sclr_X, best_model_name, best_model, best_MAPE)
                
        return best_model_name, best_bo_model, best_bo_MAPE, std_sclr               
#endregion    