import os
import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso, LassoCV
from sklearn import ensemble
from sklearn.preprocessing import StandardScaler
from kneed import KneeLocator

class FeatureSelecter:

    def set_column_info(self, df, yInput):
        if yInput == 0:
            self.response_col_name = ["Y"]
        elif yInput == 1:
            self.response_col_name = df.columns[df.columns.str.contains("Ylabel_")].to_list()
        self.dummy_factor_ls = [x for x in df.columns.values.tolist() if x not in self.response_col_name]

    def df_transform_type_all_obj(self, df):
        for col_name in df.columns.values.tolist():
            df[col_name] = df[col_name].astype('object')
        return df

    def lasso_summary(self, dummy_factor_df, dummy_response_df):
        """
        Run Lasso function and output key operation id and tool id ranking.
        """
        X, y = dummy_factor_df.values, dummy_response_df.values
        scaler = StandardScaler()
        y_scale = scaler.fit_transform(y.reshape(-1, 1))
        ### use Lasso with cross validation to choose the best parameter: alpaha instead of Lasso
        lasso_model = LassoCV(positive=True, max_iter=10000, fit_intercept=False) # Lasso Model
        lasso_model.fit(X, y_scale)
        df_ouput = pd.DataFrame(columns=["Method", "OPID_TOOLID", "IndexValue"]) # Declare df_ouput
        lasso_model_coef = lasso_model.coef_
        
        ### refit model if all coef. equal zeroes
        if all(lasso_model_coef==0):
            lasso_model = LassoCV(max_iter=10000) # Lasso Model
            lasso_model.fit(X, y_scale)
            df_ouput = pd.DataFrame(columns=["Method", "OPID_TOOLID", "IndexValue"]) # Declare df_ouput
            lasso_model_coef = lasso_model.coef_
            
        idx, names = 0, list(dummy_factor_df.columns) 
        for feature_abs_lasso_value in sorted(zip(map(lambda x: np.round(x, 4), lasso_model_coef), names), reverse=True, key=lambda x:abs(x[0])):
            if feature_abs_lasso_value[0]==0: 
                break
            df_ouput.loc[idx, :] = ["Lasso", feature_abs_lasso_value[1], feature_abs_lasso_value[0]]
            idx+=1
        return df_ouput

    def rf_summary(self, dummy_factor_df, dummy_response_df, yInput):
        """
        Run Random Forest function and output key operation id and tool id ranking.
        """
        X = np.array(dummy_factor_df)
        y = np.array(dummy_response_df)
        ### yInput correspond to the inputs from the web
        if yInput == 0:
            y = y.astype("int")
            forest = ensemble.RandomForestClassifier(n_estimators = 100,random_state = 11850)
        elif yInput == 1:
            forest = ensemble.RandomForestRegressor(n_estimators = 100,random_state = 11850)
        forest_fit = forest.fit(X, y)
        df_output = pd.DataFrame(columns=["Method", "OPID_TOOLID", "IndexValue"]) # Declare df_output
        idx = 0
        for ele in sorted(zip(map(lambda x: round(x, 4), forest_fit.feature_importances_), list(dummy_factor_df)), reverse=True, key=lambda x:abs(x[0])):
            if ele[0]==0: break
            df_output.loc[idx, :] = ["RF", ele[1], ele[0]]
            idx+=1
        return df_output

    def kneedle_cut(self, ls_label, ls_value):
        """
        Run kneedle function.
        """
        try:
            ls_idx = list(range(1, len(ls_value)+1))
            kn = KneeLocator(ls_idx, ls_value, curve="convex", direction="decreasing")
            ls_kneedle_result = ls_label[:kn.knee]
        except:
            ls_kneedle_result = ls_label
        return ls_kneedle_result

    def kneedle_optoolidresult_calcu(self, df_ls):
        """
        Output intersection of freqency of Lasso and RF results.
        """
        df_result = pd.DataFrame(columns=["Method", "OPID_TOOLID", "IndexValue"]) ### declare df_result
        row_dict = {} ### dict of op_tool_id_name
        for df in df_ls:
            Kresult_ls = df["KneedleResult"].values.tolist()
            for factor_name in Kresult_ls:
                op_tool_id_name = factor_name
                if op_tool_id_name not in df_result["OPID_TOOLID"].values.tolist():
                    ### Add row
                    row_dict[op_tool_id_name] = df_result.shape[0]
                    df_result.loc[df_result.shape[0], :] = ["Kneedle", op_tool_id_name, 1] 
                else:
                    ### Find row and add freq
                    freq_idx = df_result.columns.get_loc("IndexValue")
                    df_result.iloc[row_dict[op_tool_id_name], freq_idx]+=1
        df_result = df_result.sort_values(by="IndexValue", ascending=False).reset_index(drop=True)
        return df_result

    def calcu_weighted_value(self, ls_value):
        try:
            weight = sum(ls_value)/len(ls_value)
        except:
            weight = 0.01
        ls_weighted_value = [x/weight for x in ls_value]
        return ls_weighted_value

    def kneedle_summary(self, df_ls):
        """
        Summarize kneedle result and cut top 20.
        """
        top_num = 20
        col_value = "IndexValue"
        col_label = "OPID_TOOLID"
        df_kn_result = pd.DataFrame(columns=["Method", "OPID_TOOLID", "Inter_count", "weight_IndexValue"])
        
        for df in df_ls:
            if not df[col_value].is_monotonic_decreasing: 
                df = df.sort_values(by=col_value, ascending=False)
            ### Calculate Weighted Value
            ls_value = df[col_value].tolist()
            ls_weighted_value = self.calcu_weighted_value(ls_value)
            
            ### Extract Top Feature
            ls_top_value = df[col_value].values[:top_num].tolist()
            ls_top_weighted_value = ls_weighted_value[:top_num]
            ls_top_label = df[col_label].values[:top_num].tolist()
            
            ### Kneedle Function
            ls_label_kn = self.kneedle_cut(ls_top_label, ls_top_value)
            ls_idx_kn = [i for i in range(len(ls_top_label)) if ls_top_label[i] in ls_label_kn]
            ls_weighted_value_kn = [ls_top_weighted_value[i] for i in ls_idx_kn]
            
            ### Collect Result
            for idx in range(len(ls_label_kn)):
                label = ls_label_kn[idx]
                weighted_value = ls_weighted_value_kn[idx]
                if label not in df_kn_result[col_label].unique():
                    df_kn_result.loc[df_kn_result.shape[0], :] = ["Kneedle", label, 1, weighted_value]
                else:
                    row_idx = df_kn_result.index[df_kn_result[col_label]==label].tolist()[0]
                    df_kn_result.loc[row_idx, "Inter_count"]+=1 
                    if weighted_value>df_kn_result.loc[row_idx, "weight_IndexValue"]:
                        df_kn_result.loc[row_idx, "weight_IndexValue"] = weighted_value
                        
        return df_kn_result

    def run(self, df, yInput): 
        self.set_column_info(df, yInput)
        df = self.df_transform_type_all_obj(df)
        
        df_result = pd.DataFrame(columns=("Method", "OPID_TOOLID", "IndexValue")) 
        df_kneedle = pd.DataFrame(columns=("Method", "OPID_TOOLID", "Inter_count", "weight_IndexValue"))         
        
        df_temp = df.loc[:, self.dummy_factor_ls].drop_duplicates(subset = self.dummy_factor_ls, keep = "first", inplace = False)
        if len(self.dummy_factor_ls) == 1 and df_temp.shape[0] == 1:
            return df_result, df_kneedle

        dummy_factor_df = pd.DataFrame(pd.get_dummies(df[self.dummy_factor_ls], drop_first=True))
        dummy_response_df = df[self.response_col_name] 
        
        ### Feature Selection
        df_output_lasso = self.lasso_summary(dummy_factor_df, dummy_response_df) ### Lasso 
        df_output_rf = self.rf_summary(dummy_factor_df, dummy_response_df, yInput) ### Random Forest
        
        ### Kneedle
        df_kneedle = self.kneedle_summary([df_output_lasso, df_output_rf])
        df_result = pd.concat([df_output_lasso, df_output_rf])

        return df_result, df_kneedle

if __name__ == "__main__":
    Tester = FeatureSelecter()
    
    file_path = os.path.dirname(os.path.abspath(__file__))
    TestData = pd.read_csv(file_path+'/df_dummy.csv')
    df_result, df_kneedle = Tester.run(df=TestData, yInput=0)
    df_result.to_csv(file_path+'/df_result.csv', index=False)
    df_kneedle.to_csv(file_path+'/df_kneedle.csv', index=False)
    print('sucess')