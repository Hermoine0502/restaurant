"""
@ Created on 2020/04/06

@ author: Chingying Yang 

@ purpose: Create the prepro_csv
    
@ structure: 
    # libraries
    # defined class
        ## FeatureSlect
    # main body
"""

# region = libraries 
import pandas as pd
import numpy as np
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, SVR, LinearSVR
from xgboost import XGBClassifier, XGBRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from kneed import KneeLocator
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import f1_score, mean_squared_error
#endregion

# region = user-defined class   
class FeatureSlect:
    """
    Do feature selection.
    """
    def __init__(self):
        """
        Calculate some statistics.
        
        Parameters
        -----
        df_norm: dataframe(pandas)
            The data after preprocessing without 'sheet_id'
        
        Returns
        -----
        result_df: dataframe(pandas)
            The results of four models including F1-Score
        best_result_by_f1[:-1]: dataframe(pandas)
            The result of the best model of four models excluding F1-Score
        best_result_by_f1['Features'].to_list()[:-1]: list
            The important features of the best model
        """
        self.__features = ""
        self.__label = ""

    def run_classifier(self,df_norm):
        """
        Compare the four models performance.
        Return the best performance(F1-Score) based on four models.
        """
        ### y_input correspond to the inputs from the web
        print('run classifier start')
        target_col = 'Y'
        ### remove duplicated columns
        df_norm = df_norm.loc[:,~df_norm.columns.duplicated()]
        self.__features = df_norm.drop(target_col, axis=1)
        self.__label = df_norm[target_col]
        ### check number of label
        x, y = self.__features, self.__label
        features_list = self.__features.columns
        ### y_input correspond to the inputs from the web
     
        dict_Y = dict(pd.value_counts(y))
        if dict_Y:
            if len(dict_Y) == 1:
                msg='Label needs at least 2 classes, but only have {} class '.format(1)
                raise ValueError(msg)
            check_label_num = list((k,v) for k, v in dict_Y.items() if v < 5)
            if check_label_num:
                msg='Label needs at least 5 samples of each classes in the data, but the class {} contains only {} sample'.format(check_label_num[0][0], check_label_num[0][1])
                raise ValueError(msg)
        else:
            msg='No samples are found'
            raise ValueError(msg)
        print('splite dataset')
        seed = 3
        test_size = 0.3
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=seed,stratify=y)
        
        ### y_input correspond to the inputs from the web
        
        neg, pos = np.bincount(y_train.astype(int))
        clfs = [LogisticRegression(penalty='l2', C=10, solver='liblinear'
                , max_iter=500, class_weight='balanced')
                , SVC(kernel='linear', class_weight='balanced')
                , XGBClassifier(random_state = 11850, scale_pos_weight=neg/pos, min_child_weight=neg/pos, reg_lambda=neg/pos)
                , RandomForestClassifier(random_state = 11850, class_weight='balanced')]
       
        
        ### store four model important feature, performance(F1-score) and return the best
        result_dict, result_df, result_score_dict = {}, pd.DataFrame(), {}
        for clf in clfs:
            model = clf
            print( clf ,'module run')
            model.fit(x_train, y_train)
            ### y_input correspond to the inputs from the web
            y_predict = model.predict(x_test)
            model_metrics = f1_score(y_test, y_predict)

            ### Kneed offline
            ### order coef./imp. by abs. value
            try: 
                ### y_input correspond to the inputs from the web
                feature_importances = model.coef_[0]
            except AttributeError:
                 feature_importances = model.feature_importances_

            ### according to the absolute value of coef.(importance)
            ### choose the important feature by kneedle
            pos_coef = np.abs(feature_importances)
            kn_y = sorted(pos_coef.tolist(), reverse = True)
            kn_x = list(range(1, len(kn_y)+1))
            kn_offline = KneeLocator(kn_x, kn_y, curve='convex', direction='decreasing', online=False)

            ### rename model name
            ### y_input correspond to the inputs from the web
        
            model_name = model.__class__.__name__

            if model_name == 'LogisticRegression':
                model_name = 'Logit'
            elif model_name in ['LinearSVR', 'SVC', 'SVR']:
                model_name = 'SVM'
            elif model_name in ['XGBClassifier', 'XGBRegressor']:
                model_name = 'XGB'
            elif model_name in ['RandomForestClassifier', 'RandomForestRegressor']:
                model_name = 'RF'
                
            model_result = pd.DataFrame({'Model':model_name, 
                                     'Features':features_list,
                                     'Inter_count':1,
                                     'weight_Value':pos_coef}).sort_values(by='weight_Value', ascending=False, ignore_index=True)
            
            ### if kneedle can not work, then the top 5 features
            ### Otherwise, choose the importance features by kneedle
            try:
                print('kneedle start')
                if kn_offline.knee != 1:
                    model_result_feature = list(model_result.iloc[:kn_offline.knee-1,:]['Features'])
                else:
                    model_result_feature = list(model_result.iloc[:5,:]['Features'])
            except TypeError: 
                model_result_feature = list(model_result.iloc[:5,:]['Features'])
                        
            add_score_rows = {'Model':model_name,
                          'Features':'Accuracy',
                          'Inter_count':1,
                          'weight_Value':model_metrics}
            model_result_1 = model_result[:len(model_result_feature)].append(add_score_rows, ignore_index=True)
            
            ### store result
            result_score_dict.update({model_name : model_metrics})
            result_dict.update({model_name : model_result_1})
            result_df = result_df.append(model_result_1)
            
        
        result_df = result_df.drop('Inter_count', axis=1)    
        result_df.rename(columns={'weight_Value': 'Value'}, inplace=True)   
        print('run regression end')

        ### return the best result according to the four models
        best_result_by_f1 = result_dict[max(result_score_dict, key=result_score_dict.get)]
        return result_df, best_result_by_f1[:-1], best_result_by_f1['Features'].to_list()[:-1]
#endregion   

    def run_regression(self,df_norm):
        """
        Compare the four models performance.
        Return the best performance(mean_squared_error) based on four models.
        """

        print('run regression start')
        ### y_input correspond to the inputs from the web
        target_col = df_norm.columns[df_norm.columns.str.contains('Ylabel_')].to_list()
        ### remove duplicated columns
        df_norm = df_norm.loc[:,~df_norm.columns.duplicated()]
        self.__features = df_norm.drop(target_col, axis=1)
        self.__label = df_norm[target_col]

        ### check number of label
        x, y = self.__features, self.__label
        features_list = self.__features.columns
        ### y_input correspond to the inputs from the web
        print('splite dataset')
        seed = 3
        test_size = 0.3
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=seed)
        
        clfs = [MultiOutputRegressor(LogisticRegression(penalty='l2', C=10, solver='liblinear', max_iter=10000, class_weight='balanced'))
                    , MultiOutputRegressor(LinearSVR())
                    , MultiOutputRegressor(XGBRegressor(random_state = 11850))
                    , MultiOutputRegressor(RandomForestRegressor(random_state = 11850))]

        
        ### store four model important feature, performance(F1-score) and return the best
        result_dict, result_df, result_score_dict = {}, pd.DataFrame(), {}
        for clf in clfs:
            model = clf
            print( clf ,'module run')
            ### y_input correspond to the inputs from the web
            for col in y_train.columns.unique():
                if len(y_train[col].unique()) == 1:
                    y_train = y_train.drop(columns=col).copy()
                    y_test = y_test.drop(columns=col).copy()
            model.fit(x_train, y_train.astype(str))
            y_predict = model.predict(x_test)
            model_metrics = mean_squared_error(y_test, y_predict)

            ### Kneed offline
            ### order coef./imp. by abs. value
            try: 
                ### y_input correspond to the inputs from the web
                feature_importances = 0
                for estimators_idx in range(0, len(model.estimators_)):
                    if model.estimator.__class__.__name__ == 'LogisticRegression':
                        feature_importances +=  model.estimators_[estimators_idx].coef_.mean(axis=0)
                    else:
                        feature_importances +=  model.estimators_[estimators_idx].coef_
                feature_importances /= len(model.estimators_)
                feature_importances = np.squeeze(feature_importances).copy()
            except AttributeError:

                ### y_input correspond to the inputs from the web
                feature_importances = 0
                for estimators_idx in range(0, len(model.estimators_)):
                    feature_importances +=  model.estimators_[estimators_idx].feature_importances_
                feature_importances /= len(model.estimators_)    
                feature_importances = np.squeeze(feature_importances).copy()

            ### according to the absolute value of coef.(importance)
            ### choose the important feature by kneedle
            pos_coef = np.abs(feature_importances)
            kn_y = sorted(pos_coef.tolist(), reverse = True)
            kn_x = list(range(1, len(kn_y)+1))
            kn_offline = KneeLocator(kn_x, kn_y, curve='convex', direction='decreasing', online=False)

            ### rename model name
            ### y_input correspond to the inputs from the web
                
            model_name = model.estimator.__class__.__name__

            if model_name == 'LogisticRegression':
                model_name = 'Logit'
            elif model_name in ['LinearSVR', 'SVC', 'SVR']:
                model_name = 'SVM'
            elif model_name in ['XGBClassifier', 'XGBRegressor']:
                model_name = 'XGB'
            elif model_name in ['RandomForestClassifier', 'RandomForestRegressor']:
                model_name = 'RF'
                
            model_result = pd.DataFrame({'Model':model_name, 
                                     'Features':features_list,
                                     'Inter_count':1,
                                     'weight_Value':pos_coef}).sort_values(by='weight_Value', ascending=False, ignore_index=True)
            
            ### if kneedle can not work, then the top 5 features
            ### Otherwise, choose the importance features by kneedle
            try:
                print('kneedle start')
                if kn_offline.knee != 1:
                    model_result_feature = list(model_result.iloc[:kn_offline.knee-1,:]['Features'])
                else:
                    model_result_feature = list(model_result.iloc[:5,:]['Features'])
            except TypeError: 
                model_result_feature = list(model_result.iloc[:5,:]['Features'])
                        
            add_score_rows = {'Model':model_name,
                          'Features':'Accuracy',
                          'Inter_count':1,
                          'weight_Value':model_metrics}
            model_result_1 = model_result[:len(model_result_feature)].append(add_score_rows, ignore_index=True)
            
            ### store result
            result_score_dict.update({model_name : model_metrics})
            result_dict.update({model_name : model_result_1})
            result_df = result_df.append(model_result_1)
            
        
        result_df = result_df.drop('Inter_count', axis=1)    
        result_df.rename(columns={'weight_Value': 'Value'}, inplace=True)  
        print('run regression end')
        ### return the best result according to the four models
        best_result_by_f1 = result_dict[max(result_score_dict, key=result_score_dict.get)]
        return result_df, best_result_by_f1[:-1], best_result_by_f1['Features'].to_list()[:-1]
#endregion   
# region = main body
if __name__ == '__main__':
    yInput = 1
    if yInput == 0:
        df_norm=pd.read_csv('D:/UserData/EllieCYYang/testdata.csv')
        FeatureSlect=FeatureSlect()
        classifier=FeatureSlect.run_classifier(df_norm)
        print(classifier)
    elif yInput == 1:
        df_norm=pd.read_csv('D:/UserData/EllieCYYang/testdataRE.csv')
        FeatureSlect=FeatureSlect()
        regression=FeatureSlect.run_regression(df_norm)
        print(regression)
        
#endregion
