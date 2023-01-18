import os
import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.stats import chi2_contingency
import time
from lib_function import df_transform_type_all_obj, df_transform_type_from_header, drop_col_with_rework_flag

class main_proportion_test:
    """
    This package first create ng ratio table, then do the chi-squared test, mark the key machines,
    and finally output the result table.
    """
    def __init__(self):
        # print("init")
        pass

    def raw_info_recording(self, df_ng_ratio_site, level_list):
        """
        Create ng ratio tables for each level of the input site dataframe for next analysis.

        Parameters
        -----
        df_ng_ratio_site: dataframe(pandas)     
            Ng ratio of op_id, line_id, eqp_id, unit_name, unit_id.
        level_list: list
            list of site level.
        
        Returns
        -----
        df_top_level_final: list of dataframe(pandas) 
            list of ng ratio table of site for each level that need to be test and add a nan column for p-value
        """
        list_len = len(level_list) 
        df_top_level_final = []
        for i in range(1, list_len):
            site_list_i = level_list[0:i+1]
            site_list_i.append("site") # use this column to double check the site is right
            site_list_i_drop = level_list[i+1:] 
            tmp_level_none = df_ng_ratio_site[df_ng_ratio_site[site_list_i[-2]]!="None"] # to check the current level that are goint to test is not 'None'    
            
            ### get the NG/Total number and NG Ratio
            sum_ng_num_tmp = tmp_level_none.drop(site_list_i_drop, axis=1).groupby(site_list_i)["NG_num"].agg("sum").reset_index()
            sum_total_num_tmp = tmp_level_none.drop(site_list_i_drop, axis = 1).groupby(site_list_i)["Total_num"].agg("sum").reset_index()
            df_top_level = pd.concat([sum_ng_num_tmp, pd.DataFrame(sum_total_num_tmp["Total_num"])], axis=1)
            print(df_top_level["NG_num"])
            print(df_top_level["Total_num"])
            df_top_level["NG_Ratio"] = round(df_top_level["NG_num"]/df_top_level["Total_num"]*100, 2)
            df_top_level["p-value"] = np.nan 
            df_top_level = df_top_level.sort_values(["Parameter","NG_Ratio"], ascending=[True, False]).reset_index(drop=True)
            df_top_level_final.append(df_top_level)
        return df_top_level_final 

    def chi_squared_test(self, df_top_level_final_i, level_list, i):
        """
        Collect the group that are going to do chi-squared test and do the test.

        Parameters
        -----
        df_top_level_final_i: dataframe(pandas)     
            dataframe of level that are going to do the chi-squared test.
        level_list: list
            ist of site level.
        i: int
            choose the i-th dataframe of the input list of do_chi_squared_test().
        
        Returns
        -----
        df_top_level_test_final_i: dataframe(pandas) 
            dataframe of level after testing.
        """
        site_list_last_level = level_list[0:i] 
        ngnum_idx = df_top_level_final_i.columns.get_loc("NG_num")
        totalnum_idx = df_top_level_final_i.columns.get_loc("Total_num")
        pvalue_idx = df_top_level_final_i.columns.get_loc("p-value")
        df_top_level_final_i["sig_flag_p_value"] = ""
        df_top_level_final_i["true_ng"] = ""  
        df_top_level_final_i["expect_ng"] = "" 
        df_top_level_final_i["sig_flag"] = ""
        
        ### collect the group that are going to do chi-squared test
        row_idx = 0
        while (row_idx < (df_top_level_final_i.shape[0]-1)):
            tmp_idx = []
            while row_idx < df_top_level_final_i.shape[0]-1 and (df_top_level_final_i.loc[row_idx, site_list_last_level] == df_top_level_final_i.loc[row_idx+1, site_list_last_level]).all():
                tmp_idx.append(row_idx)
                tmp_idx.append(row_idx + 1)
                row_idx = row_idx + 1
            if len(tmp_idx) > 1:
                tmp_idx = list(set(tmp_idx))
                tmp_df = np.array(df_top_level_final_i.iloc[tmp_idx, [ngnum_idx, totalnum_idx]])
                tmp_df[:,1] = tmp_df[:,1] - tmp_df[:,0]
                if not (tmp_df == 0).any():
                    
                    ### chi-squared test
                    p_value = chi2_contingency(tmp_df)[1] 
                    expect_num = chi2_contingency(tmp_df)[3]   
                    if p_value < 0.05: 
                        df_top_level_final_i.iloc[tmp_idx, pvalue_idx] = p_value
                        df_top_level_final_i.loc[tmp_idx, "sig_flag_p_value"] = "Y"
                        for i in range(len(expect_num)):
                            tmp_ng_num = tmp_df[i][0]
                            tmp_total_num = tmp_df[i][0] + tmp_df[i][1]
                            tmp_ng_ratio = tmp_ng_num / tmp_total_num
                            tmp_ng_num_e = expect_num[i][0]
                            tmp_total_num_e = expect_num[i][0] + expect_num[i][1]
                            tmp_ng_ratio_e = tmp_ng_num_e / tmp_total_num_e
                            df_top_level_final_i.loc[tmp_idx[i], "true_ng"] = tmp_ng_ratio
                            df_top_level_final_i.loc[tmp_idx[i], "expect_ng"] = tmp_ng_ratio_e
                            if tmp_ng_ratio > tmp_ng_ratio_e:
                                df_top_level_final_i.loc[tmp_idx[i], "sig_flag"] = "Y"
            else:
                row_idx = row_idx + 1
                continue 
        df_top_level_test_final_i = df_top_level_final_i
        return df_top_level_test_final_i

    def do_chi_squared_test(self, df_top_level_final, level_list):
        """
        Do chi-squared test for each level of the input site dataframe list.

        Parameters
        -----
        df_top_level_final: list of dataframe(pandas)   
            list of ng ratio table of site for each level that need to be test.
        level_list: list
            list of site level.
        
        Returns
        -----
        df_top_level_test_final: list of dataframe(pandas)    
            list of dataframe of level after testing.
        """
        list_len = len(level_list)
        df_top_level_test_final = []
        for i in range(1, list_len):
            df_top_level_final_i = df_top_level_final[i-1] 
            print('-'*15)
            print('test level:', level_list[i])
            start = time.time()
            df_top_level_test_final_i = self.chi_squared_test(df_top_level_final_i, level_list, i)
            end = time.time()
            time_diff = end-start
            print('time cost:', time_diff)
            df_top_level_test_final.append(df_top_level_test_final_i)
        return df_top_level_test_final

    def flag_and_drop(self, df_top_level_test_final, level_list):
        """
        Set the level flag and drop the useless columns.

        Parameters
        -----
        df_top_level_test_final: list of dataframe(pandas)    
            list of dataframe of level after testing.
        level_list: list
            list of site level.
        
        Returns
        -----
        df_top_level_test_flag_final: list of dataframe(pandas)    
            list of dataframe of level after flagging and dropping.
        """
        list_len = len(level_list)
        df_top_level_test_flag_final = []
        for i in range(1, list_len): 
            site_list_i = level_list[0:i+1]
            df_top_level_test_final_i = df_top_level_test_final[i-1]
            df_top_level_test_final_i = df_top_level_test_final_i.drop(["sig_flag_p_value", "true_ng", "expect_ng"], axis=1)
            df_top_level_test_final_i["flag"] = site_list_i[-1]
            df_top_level_test_flag_final.append(df_top_level_test_final_i)
        return df_top_level_test_flag_final

    def run(self, df_ng_ratio_site, site_level_list):
        """
        Run chi-squared test to find the key machines, then concat the required information to form the final table.

        Parameters
        -----
        df_ng_ratio_site: dataframe(pandas)     
            Ng ratio of op_id, line_id, eqp_id, unit_name, unit_id.
        site_level_list: list
            list of site level.
        
        Returns
        -----
        df_output_proportion: dataframe(pandas) 
            result of chi-squared test correspond to each level, including the required information.
            
        """
        if not isinstance(df_ng_ratio_site, pd.DataFrame): 
            raise TypeError('Input is not a dataframe.')
        response_column_name = "Y"

        indicator_name = ["flag", "site", "NG_num", "Total_num", "NG_Ratio", "p-value", "sig_flag"]
        col_name = site_level_list + indicator_name
        df_output_proportion = pd.DataFrame(columns=col_name)
        if not df_ng_ratio_site.empty:
            print('Site:', df_ng_ratio_site['site'].unique()[0])
            df_level_final = self.raw_info_recording(df_ng_ratio_site, site_level_list)
            df_level_test_final = self.do_chi_squared_test(df_level_final, site_level_list)
            df_level_test_flag_final = self.flag_and_drop(df_level_test_final, site_level_list)
            df_final = pd.concat(df_level_test_flag_final, axis=0)
            df_output_proportion = pd.concat([df_output_proportion, df_final], axis=0)   
        return df_output_proportion

if __name__ == '__main__':
    path = './test_data/'
    dirs_list = os.listdir(path)
    
    ### exapmle
    file = dirs_list[0]
    df_ng_ratio = pd.read_csv(path + file)
    df_ng_ratio = df_transform_type_all_obj(df_ng_ratio)
    df_ng_ratio = drop_col_with_rework_flag(df_ng_ratio)
    df_ng_ratio.rename(columns = {"op_id_info":"Parameter", "ng_count":"NG_num", "all_count":"Total_num"}, inplace=True) # rename columns
    
    # Set level list of each site 
    array_level_list = ["Parameter", "tool_id", "chmb"]
    cf_level_list = ["Parameter", "line_id", "tool_id"]
    cell_level_list = ["Parameter", "tool_id", "unit_name","chmb"]
    
    # Split df_ng_ratio by site 
    df_ng_ratio_array = df_ng_ratio[df_ng_ratio["site"] == "ARRAY"]
    df_ng_ratio_cf = df_ng_ratio[df_ng_ratio["site"] == "CF"]
    df_ng_ratio_cell = df_ng_ratio[df_ng_ratio["site"] == "CELL"]
    
    df_output_proportion = pd.DataFrame(columns=(["Parameter", "line_id", "tool_id", "unit_name", "chmb", "flag", "site", "NG_num", "Total_num", "NG_Ratio", "p-value", "sig_flag"]))
    main_proportion_array = main_proportion_test()
    df_output_proportion_array = main_proportion_array.run(df_ng_ratio_array, array_level_list)
    
    main_proportion_cf = main_proportion_test()
    df_output_proportion_cf = main_proportion_cf.run(df_ng_ratio_cf, cf_level_list)
    
    main_proportion_cell = main_proportion_test()
    df_output_proportion_cell = main_proportion_cell.run(df_ng_ratio_cell, cell_level_list)
    
    df_output_proportion = pd.concat([df_output_proportion, df_output_proportion_array, df_output_proportion_cf, df_output_proportion_cell], axis=0)
    df_output_proportion.to_csv('df_output_proportion.csv',index=False)