def run(df_norm, df_unnorm, yInput):
    """
    Compare the four models performance.
    Return the best performance(F1-Score) based on four models.
    
    Parameters
    -----
    df_norm: dataframe(pandas)
        The data after preprocessing without 'sheet_id'
    df_unnorm: dataframe(pandas) 
        The data after preprocessing with 'sheet_id'
        
    Returns
    -----
    result_df: dataframe(pandas)
        The results of four models including F1-Score
    best_result_by_f1[:-1]: dataframe(pandas)
        The result of the best model of four models excluding F1-Score
    best_result_by_f1['Features'].to_list()[:-1]: list
        The important features of the best model
    """

    ### y_input correspond to the inputs from the web
    if yInput == 0:
        target_col = 'Y'
    elif yInput == 1:
        target_col = df_norm.columns[df_norm.columns.str.contains('Ylabel_')].to_list()
        
    ### remove duplicated columns
    df_norm = df_norm.loc[:,~df_norm.columns.duplicated()]
    feature_df = df_norm.drop(target_col, axis=1)
    target_df = df_norm[target_col]
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
    # from hyperopt import fmin, tpe, hp, Trials ### Bayesian Optimization
    # from hyperopt.early_stop import no_progress_loss
    
    ### check number of label
    x, y = feature_df, target_df
    ### y_input correspond to the inputs from the web
    if yInput == 0:
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

    seed = 3
    test_size = 0.3
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=seed)
    
    ### y_input correspond to the inputs from the web
    if yInput == 0:
        neg, pos = np.bincount(y_train.astype(int))
        clfs = [LogisticRegression(penalty='l2', C=10, solver='liblinear'
                , max_iter=500, class_weight='balanced')
                , SVC(kernel='linear', class_weight='balanced')
                , XGBClassifier(random_state = 11850, scale_pos_weight=neg/pos, min_child_weight=neg/pos, reg_lambda=neg/pos)
                , RandomForestClassifier(random_state = 11850, class_weight='balanced')]
    elif yInput == 1:
        clfs = [MultiOutputRegressor(LogisticRegression(penalty='l2', C=10, solver='liblinear', max_iter=500, class_weight='balanced'))
                , MultiOutputRegressor(LinearSVR())
                , MultiOutputRegressor(XGBRegressor(random_state = 11850))
                , MultiOutputRegressor(RandomForestRegressor(random_state = 11850))]

    
    ### store four model important feature, performance(F1-score) and return the best
    result_dict, result_df, result_score_dict = {}, pd.DataFrame(), {}
    for clf in clfs:
#        def hyperopt_model_score(params, clf=clf, cv=cv, X=x_train, y=y_train):
#            """
#            The function gets a set of parameters in "param"
#            Use this params to create a new model
#            and then conduct the cross validation with the same folds as before
#            """
#            return -cross_val_score(clf(**params), X, y, cv=cv, scoring='f1').mean()
#        ### trials will contain logging information
#        trials = Trials()
#        best = fmin(fn=hyperopt_model_score, ### function to optimize
#                       space=para, 
#                       algo=tpe.suggest, ### optimization algorithm, hyperotp will select its parameters automatically
#                       max_evals=n_iter, ### maximum number of iterations
#                       trials=trials, ### logging
#                       rstate=np.random.RandomState(seed), ### fixing random state for the reproducibility
#                       early_stop_fn=no_progress_loss(percent_increase=5)) ### early stopping
#        ### update the para. searched by Bayesian opt
#        ### retrain the model by the best para.
#        para.update(best)
        model = clf
        ### y_input correspond to the inputs from the web
        if yInput == 0:
            model.fit(x_train, y_train)
        elif yInput == 1:
            for col in y_train.columns.unique():
                if len(y_train[col].unique()) == 1:
                   y_train = y_train.drop(columns=col).copy()
                   y_test = y_test.drop(columns=col).copy()
            model.fit(x_train, y_train.astype(str))
        y_predict = model.predict(x_test)
         
        if yInput == 0:
            model_metrics = f1_score(y_test, y_predict)
        elif yInput == 1:
            model_metrics = mean_squared_error(y_test, y_predict)

        ### Kneed offline
        ### order coef./imp. by abs. value
        try: 
            ### y_input correspond to the inputs from the web
            if yInput == 0:
                feature_importances = model.coef_[0]
            elif yInput == 1:
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
            if yInput == 0:
                feature_importances = model.feature_importances_
            elif yInput == 1:
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
        if yInput == 0:
            model_name = model.__class__.__name__
        elif yInput == 1:            
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
                                     'Features':feature_df.columns,
                                     'Inter_count':1,
                                     'weight_Value':pos_coef}).sort_values(by='weight_Value', ascending=False, ignore_index=True)
        
        ### if kneedle can not work, then the top 5 features
        ### Otherwise, choose the importance features by kneedle
        try:
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
    
    ### return the best result according to the four models
    best_result_by_f1 = result_dict[max(result_score_dict, key=result_score_dict.get)]
    return result_df, best_result_by_f1[:-1], best_result_by_f1['Features'].to_list()[:-1]