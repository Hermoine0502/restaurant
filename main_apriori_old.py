import os
from apyori import apriori
import numpy as np
import pandas as pd
from lib_function import df_transform_type_from_header, df_transform_type_all_obj, drop_col_with_rework_flag

def run(df):
    """
    Run association rule function to find the accociation between tools, we will partition dataframe into two parts, Y=0 and Y=1.
    And calculate ng ratio respectively.
    
    Parameters
    -----
    df: dataframe(pandas)
        df_phaseI.
    
    Returns
    -----
    association_rule: dataframe(pandas)
        Association rule.
    
    """
    ### Create two dataframe for Y=0 or Y=1
    ### Partition into two dataframe for Y=0 or Y=1
    response_col_name = 'Y'
    df = df_transform_type_all_obj(df)
    df = drop_col_with_rework_flag(df)
    df_ori = df.copy()
    df_0 = df[df[response_col_name]==0]
    df_1 = df[df[response_col_name]==1]
    df_0 = df_0.drop(response_col_name, axis=1)
    df_1 = df_1.drop(response_col_name, axis=1)

    ### Replace cell_value with col_name+cell_value that means cell_value will be opid+toolid
    for col_idx in range(df_0.shape[1]):
        col_name = list(df_0)[col_idx]
        for row_idx in range(df_0.shape[0]):
            cell_value = df_0.iloc[row_idx, col_idx]
            new_value = col_name+'='+str(cell_value)
            df_0.iloc[row_idx, col_idx] = new_value
            
    for col_idx in range(df_1.shape[1]):
        col_name = list(df_1)[col_idx]
        for row_idx in range(df_1.shape[0]):
            cell_value = df_1.iloc[row_idx, col_idx]
            new_value = col_name+'='+str(cell_value)
            df_1.iloc[row_idx, col_idx] = new_value
            
    for col_idx in range(df_ori.shape[1]):
        col_name = list(df_ori)[col_idx]
        for row_idx in range(df_ori.shape[0]):
            cell_value = df_ori.iloc[row_idx, col_idx]
            new_value = col_name+'='+str(cell_value)
            df_ori.iloc[row_idx, col_idx] = new_value

    ### Define apriori function parameters and execute it (Y=0)
    min_support_value = 0.5
    min_confidence_value = 0.5
    min_lift_value = 1.000001
    max_length_value = 3
    association_rules_0 = apriori(np.array(df_0), min_support = min_support_value, min_confidence = min_confidence_value, min_lift = min_lift_value, max_length = max_length_value)
    association_results_0 = list(association_rules_0)

    ### Remove rules if op_id = 0 (Y=0)
    association_df = pd.DataFrame(columns=('X1','X2','X3','Support','Confidence','Lift')) 
    Antecedent_ls, Antecedent_ls1, Consequent_ls, Support_ls, Confidence_ls, Lift_ls  = [], [], [], [], [], []
    for RelationRecord in association_results_0:
        for ordered_stat in RelationRecord.ordered_statistics:
            if len(list(ordered_stat.items_base)) > 0:
                try:
                    if list(ordered_stat.items_base)[0].split('=')[1]=='0' or list(ordered_stat.items_base)[1].split('=')[1]=='0' or list(ordered_stat.items_add)[0].split('=')[1]=='0':
                        continue
                except:
                    if list(ordered_stat.items_base)[0].split('=')[1]=='0' or list(ordered_stat.items_add)[0].split('=')[1]=='0':
                        continue
                Antecedent_ls.append(list(ordered_stat.items_base)[0])
                try:
                    Antecedent_ls1.append(list(ordered_stat.items_base)[1])
                except:
                    Antecedent_ls1.append('')
                Consequent_ls.append(list(ordered_stat.items_add)[0])
                Support_ls.append(round(RelationRecord.support,4))
                Confidence_ls.append(round(ordered_stat.confidence,4))
                Lift_ls.append(round(ordered_stat.lift,4))
    association_df['X1'], association_df['X2'], association_df['X3'], association_df['Support'], association_df['Confidence'], association_df['Lift'] = Antecedent_ls, Antecedent_ls1, Consequent_ls, Support_ls, Confidence_ls, Lift_ls
    association_df = association_df.sort_values(['Lift', 'Confidence'], ascending=False).reset_index(drop=True) ### Sorted by 'Lift', 'Confidence'

    association_df = pd.DataFrame.drop_duplicates(association_df).reset_index(drop=True)
    
    ### Remove duplicate rules (Y=0)
    remove_idx = []
    for i in range(association_df.shape[0]):
        j = i + 1
        if j < association_df.shape[0]:
            if ( association_df['X1'][i] == association_df['X1'][j] and 
                association_df['X2'][i] == association_df['X2'][j] and 
                association_df['X3'][i] == association_df['X3'][j]):
                remove_idx.append(j)   
    association_df = association_df.drop(remove_idx, axis=0)
    
    ### Create original rule list, in order to compare to association rule results (Y=0)
    ls2 = []    
    for rule_ori in range(df_ori.shape[0]):
        ls2.append(df_ori.iloc[rule_ori].tolist())
        ''' rocky test
        ls_temp = []
        for idx in range(len(df_ori.iloc[rule_ori,:])):
            ls_temp.append(df_ori.iloc[rule_ori,:][idx])
        ls2.append(ls_temp)'''
    
    ### Calculate ng ratio (Y=0) 
    ng_ratio_ls = []
    for rule_idx in range(association_df.shape[0]):
        ls1 = []
        ls1.append(association_df.iloc[rule_idx,:]['X1'])
        if association_df.iloc[rule_idx,:]['X2'] != '':
            ls1.append(association_df.iloc[rule_idx,:]['X2'])
        ls1.append(association_df.iloc[rule_idx,:]['X3'])
        all_count = 0
        ng_count = 0
        for check in range(len(ls2)):
            if all(item in ls2[check] for item in ls1):
                all_count = all_count + 1
            if all(item in ls2[check] for item in ls1) and ls2[check][0] == 'Y=1':
                ng_count = ng_count + 1
        ngratio = ng_count/all_count
        ng_ratio_ls.append(ngratio)
        
    association_df['NG_Ratio'] = ng_ratio_ls
    try:  
        association_df1 = pd.concat([association_df[association_df['X2'] != ''].sort_values(by='NG_Ratio'),association_df[association_df['X2'] == ''].sort_values(by='NG_Ratio')], axis=0)
        association_df1 = association_df1.loc[:,['X1','X2','X3','Lift','NG_Ratio']].reset_index(drop=True)
        
        remove_idx = []
        for i in range(association_df1.shape[0]):
            j = i + 1
            if j < association_df.shape[0]:
                if association_df1['NG_Ratio'][i] == association_df1['NG_Ratio'][j]:
                    remove_idx.append(j)   
        association_rule_0 = association_df1.drop(remove_idx, axis=0)
    except:
        association_rule_0 = association_df
        
    ### Run apriori function (Y=1)
    association_rules_1 = apriori(np.array(df_1), min_support = min_support_value, min_confidence = min_confidence_value, min_lift = min_lift_value, max_length = max_length_value)
    association_results_1 = list(association_rules_1)

    ### Remove rules if op_id = 0 (Y=1)
    association_df = pd.DataFrame(columns=('X1','X2','X3','Support','Confidence','Lift')) 
    Antecedent_ls, Antecedent_ls1, Consequent_ls, Support_ls, Confidence_ls, Lift_ls  = [], [], [], [], [], []
    for RelationRecord in association_results_1:
        for ordered_stat in RelationRecord.ordered_statistics:
            if len(list(ordered_stat.items_base)) > 0:
                try:
                    if list(ordered_stat.items_base)[0].split('=')[1]=='0' or list(ordered_stat.items_base)[1].split('=')[1]=='0' or list(ordered_stat.items_add)[0].split('=')[1]=='0':
                        continue
                except:
                    if list(ordered_stat.items_base)[0].split('=')[1]=='0' or list(ordered_stat.items_add)[0].split('=')[1]=='0':
                        continue
                Antecedent_ls.append(list(ordered_stat.items_base)[0])
                try:
                    Antecedent_ls1.append(list(ordered_stat.items_base)[1])
                except:
                    Antecedent_ls1.append('')
                Consequent_ls.append(list(ordered_stat.items_add)[0])
                Support_ls.append(round(RelationRecord.support,4))
                Confidence_ls.append(round(ordered_stat.confidence,4))
                Lift_ls.append(round(ordered_stat.lift,4))
    association_df['X1'], association_df['X2'], association_df['X3'], association_df['Support'], association_df['Confidence'], association_df['Lift'] = Antecedent_ls, Antecedent_ls1, Consequent_ls, Support_ls, Confidence_ls, Lift_ls
    association_df = association_df.sort_values(['Lift', 'Confidence'], ascending=False).reset_index(drop=True) ### Sorted by 'Lift', 'Confidence'

    association_df = pd.DataFrame.drop_duplicates(association_df).reset_index(drop=True)
    
    ### Remove duplicate rules (Y=1)
    remove_idx = []
    for i in range(association_df.shape[0]):
        j = i + 1
        if j < association_df.shape[0]:
            if association_df['X1'][i] == association_df['X1'][j] and association_df['X2'][i] == association_df['X2'][j] and association_df['X3'][i] == association_df['X3'][j]:
                remove_idx.append(j)   
    association_df = association_df.drop(remove_idx, axis=0)
    
    ### Create original rule list, in order to compare to association rule results (Y=1)
    ls2 = []    
    for rule_ori in range(df_ori.shape[0]):
        ls_temp = []
        for idx in range(len(df_ori.iloc[rule_ori,:])):
            ls_temp.append(df_ori.iloc[rule_ori,:][idx])
        ls2.append(ls_temp)
    
    ### Calculate ng ratio (Y=1)
    ng_ratio_ls = []
    for rule_idx in range(association_df.shape[0]):
        ls1 = []
        ls1.append(association_df.iloc[rule_idx,:]['X1'])
        if association_df.iloc[rule_idx,:]['X2'] != '':
            ls1.append(association_df.iloc[rule_idx,:]['X2'])
        ls1.append(association_df.iloc[rule_idx,:]['X3'])
        all_count = 0
        ng_count = 0
        for check in range(len(ls2)):
            if all(item in ls2[check] for item in ls1):
                all_count = all_count + 1
            if all(item in ls2[check] for item in ls1) and ls2[check][0] == 'Y=1':
                ng_count = ng_count + 1
        ngratio = ng_count/all_count
        ng_ratio_ls.append(ngratio)
        
    association_df['NG_Ratio'] = ng_ratio_ls
    try:  
        association_df1 = pd.concat([association_df[association_df['X2'] != ''].sort_values(by='NG_Ratio'),association_df[association_df['X2'] == ''].sort_values(by='NG_Ratio')], axis=0)
        association_df1 = association_df1.loc[:,['X1','X2','X3','Lift','NG_Ratio']].reset_index(drop=True)
        
        remove_idx = []
        for i in range(association_df1.shape[0]):
            j = i + 1
            if j < association_df.shape[0]:
                if association_df1['NG_Ratio'][i] == association_df1['NG_Ratio'][j]:
                    remove_idx.append(j)   
        association_rule_1 = association_df1.drop(remove_idx, axis=0)
    except:
        association_rule_1 = association_df
        
    association_rule_0['Y'] = 0
    association_rule_1['Y'] = 1
    
    ### Combine Y=0 and Y=1 results
    association_rule = pd.concat([association_rule_0,association_rule_1],axis=0)
    
    return association_rule

