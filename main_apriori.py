from apyori import apriori
import numpy as np
import pandas as pd
from lib_function import df_transform_type_all_obj, drop_col_with_rework_flag
from ast import literal_eval
from operator import index
import main_apriori_old
import time
import os
    
class apriori_algorithm:

    def __init__(self, min_support=0.5, min_confidence=0.5, min_lift=1.000001, max_length=3):
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.min_lift = min_lift
        self.max_length = max_length

    def revise_cell_value(self, data):
        """
        Replace cell_value with col_name+cell_value that means cell_value will be opid+toolid.
        
        Parameters
        -----
        data: dataframe(pandas)
            df_0 or df_1 or df_ori.
        
        Returns
        -----
        data: dataframe(pandas)
            Result which after replac
        """
        for col_idx in range(data.shape[1]):
            col_name = list(data)[col_idx]
            for row_idx in range(data.shape[0]):
                cell_value = data.at[row_idx, col_name]
                new_value = col_name+"="+str(cell_value)
                data.at[row_idx, col_name] = new_value
        return data

    def main_process(self ,data, ls2):
        """
        Replace cell_value with col_name+cell_value that means cell_value will be opid+toolid.
        
        Parameters
        -----
        data: dataframe(pandas)
            df_0 or df_1.
        ls2: list
            Original rule list.
        Returns
        -----
        association_rule: dataframe(pandas)
            Association rule for data.
        """
        association_results = list(apriori(np.array(data), min_support=self.min_support, min_confidence=self.min_confidence, min_lift=self.min_lift, max_length=self.max_length))
        ### Remove rules if op_id = 0 (Y=0)
        association_df = pd.DataFrame(columns=("item_base","item_add","Support","Confidence","Lift")) 
        item_base_ls, item_add_ls, Support_ls, Confidence_ls, Lift_ls, item_ls  = [], [], [], [], [], []
        for RelationRecord in association_results:
            item_set = list(RelationRecord.items)
            if True in ["=0" in item for item in item_set] :
                continue
            for ordered_stat in RelationRecord.ordered_statistics:
                if len(list(ordered_stat.items_base)) > 0:
                    item_list = list(ordered_stat.items_base) + list(ordered_stat.items_add)                    
                    item_base_ls.append(list(ordered_stat.items_base))
                    item_add_ls.append(list(ordered_stat.items_add))
                    Support_ls.append(round(RelationRecord.support,4))
                    Confidence_ls.append(round(ordered_stat.confidence,4))
                    Lift_ls.append(round(ordered_stat.lift,4))
                    item_ls.append(str(sorted(item_list)))
        association_df["item_base"], association_df["item_add"], association_df["Support"], association_df["Confidence"], association_df["Lift"] = item_base_ls, item_add_ls, Support_ls, Confidence_ls, Lift_ls
        association_df["item"] = item_ls
        association_df = association_df.sort_values(["Lift", "Confidence"], ascending=False).reset_index(drop=True) ### Sorted by "Lift", "Confidence"
        ### Calculate ng ratio (Y=0) 
        ng_ratio_ls = []
        for rule_idx in range(association_df.shape[0]):
            ls1 = literal_eval(association_df.at[rule_idx, "item"])            
            all_count = 0
            ng_count = 0            
            for row in ls2:
                if all(item in row for item in ls1):
                    all_count += 1
                    if row[0] == "Y=1":
                        ng_count += 1
            ngratio = ng_count/all_count
            ng_ratio_ls.append(ngratio)
            
        association_df["NG_Ratio"] = ng_ratio_ls
        
        association_df1 = association_df.sort_values(by="NG_Ratio")
        association_df1 = association_df1.loc[:,["item_base","item_add","Lift","NG_Ratio"]].reset_index(drop=True)
        
        return association_df1

    def run(self, df, response_col_name="Y"):
        """
        Run association rule function to find the accociation between tools, we will partition dataframe into two parts, Y=0 and Y=1.
        And calculate ng ratio respectively.
        
        Parameters
        -----
        df: dataframe(pandas)
            df_phaseI.
        response_col_name : str
            target's column name
        
        Returns
        -----
        association_rule: dataframe(pandas)
            Association rule.
        """
        ### Create two dataframe for Y=0 or Y=1
        ### Partition into two dataframe for Y=0 or Y=1
        assert isinstance(df, pd.DataFrame), "input data is not a dataframe"
        df = df_transform_type_all_obj(df)
        df = drop_col_with_rework_flag(df)
        df_ori = df.copy()
        df_0 = df[df[response_col_name]==0].reset_index(drop=True)
        df_1 = df[df[response_col_name]==1].reset_index(drop=True)
        df_0 = df_0.drop(response_col_name, axis=1)
        df_1 = df_1.drop(response_col_name, axis=1)

        ### Replace cell_value with col_name+cell_value that means cell_value will be opid+toolid
        df_0 = self.revise_cell_value(df_0)
        df_1 = self.revise_cell_value(df_1)
        df_ori = self.revise_cell_value(df_ori)

        ### Create original rule list, in order to compare to association rule results
        ls2 = []    
        for rule_ori in range(df_ori.shape[0]):
            ls2.append(df_ori.iloc[rule_ori].tolist())
        
        association_rule_0 = self.main_process(df_0, ls2)
        association_rule_1 = self.main_process(df_1, ls2)
        
        association_rule_0["Y"] = 0
        association_rule_1["Y"] = 1
        
        ### Combine Y=0 and Y=1 results
        association_rule = pd.concat([association_rule_0,association_rule_1],axis=0)
        association_rule["item_base"] = association_rule["item_base"].apply(lambda x: ",".join(map(str, x)))
        association_rule["item_add"] = association_rule["item_add"].apply(lambda x: ",".join(map(str, x)))
        return association_rule
    
if __name__=="__main__":
    path = './apriori_test_data/'
    dirs = os.listdir(path)
    df_use_time = pd.DataFrame()
    file_list = []
    before_use_time_list = []
    after_use_time_list = []
    compare_list = []
    for file in dirs:
        file_list.append(file)
        df_phaseI = pd.read_csv(f'{path}{file}')
        # start = time.time()
        # result = main_apriori_old.run(df_phaseI)
        # end = time.time()
        # before = end-start
        # before_use_time_list.append(round(before,3))
        # start = time.time()
        apriori_alg = apriori_algorithm()
        result = apriori_alg.run(df_phaseI)
        # end = time.time()
        # after = end-start
        # after_use_time_list.append(round(after,3))
        # change = round((after-before)/before,3)*100
        # compare_list.append(str(change) + '%')
        result.to_csv('result.csv',index=False)



    # df_use_time['file']= file_list
    # df_use_time['before']= before_use_time_list
    # df_use_time['after']= after_use_time_list
    # df_use_time['compare']= compare_list
    # df_use_time.to_csv('use_time.csv',index=False)