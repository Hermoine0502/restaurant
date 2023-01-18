import pandas as pd
import numpy as np
import pickle

def remove_col_with_same_value(df):
    same_value_col_ls = []
    df_des = df.describe()
    for col_name in df_des.columns:
        if df_des.loc['min',col_name] == df_des.loc['max',col_name]:
            same_value_col_ls.append(col_name)
    df = df.drop(same_value_col_ls, axis=1)
    return df

def remove_col_with_same_cate(df):
    same_cate_col_ls = []
    for col_name in df.columns:
        cate_dict = df.groupby(col_name).groups
        if len(cate_dict) == 1:
            same_cate_col_ls.append(col_name)
    df = df.drop(same_cate_col_ls, axis=1)
    return df

def remove_col_with_unique_value(df):
    nunique_ser = df.apply(pd.Series.nunique)
    drop_col_name_ls = nunique_ser[nunique_ser==1].index.values.tolist()
    df = df.drop(drop_col_name_ls,axis=1)
    return df

def replace_missing_value_by_column_mode(df):
    for col_idx in range(df.shape[1]):
        col_name = df.columns.values.tolist()[col_idx]
        if df[col_name].hasnans:
            column_mode_value = df.mode()[col_name][0]
            for row_idx in range(df.shape[0]):
                cell_value = df.iloc[row_idx, col_idx]
                if pd.isnull(cell_value):
                    df.iloc[row_idx, col_idx] = column_mode_value
                elif cell_value == '':
                    df.iloc[row_idx, col_idx] = column_mode_value
    return df

def replace_missing_value_by_zero(df):
    df = df.fillna(0)
    return df
            
def remove_row_duplicate(df): 
    df = df.drop_duplicates()
    return df

def df_transform_type_from_header(df):
    type_ls = df.columns.values.tolist() # Get content_type list
    for type_idx in range(len(type_ls)):
        type_str = type_ls[type_idx]
        if '.' in type_str:
            type_ls[type_idx] = type_str.split('.')[0]
    df.columns = df.iloc[0,:] # Get para_name list
    df = df.drop(df.index[[0]]) # Delete first row

    for col_idx in range(df.shape[1]):
        col_name =  df.columns.values.tolist()[col_idx]
        if type_ls[col_idx] == 'categorical':
            df[col_name] = df[col_name].astype('object')
        elif type_ls[col_idx] == 'continuous':
            df[col_name] = df[col_name].astype('float')
    return df

def df_transform_type_all_obj(df):
    for col_name in df.columns.values.tolist():
        df[col_name] = df[col_name].astype('object')
    return df

def add_stationdt_type_header(df, Y_type_str=False):
    raw_header_ls = df.columns.values.tolist()
    new_header_ls = []
    for name in raw_header_ls:
        if Y_type_str and name == 'Y':
            new_header_ls.append(Y_type_str)
        else:
            new_header_ls.append('categorical')
    df.columns = new_header_ls
    raw_header_df = pd.DataFrame(np.array([raw_header_ls]), columns=new_header_ls)
    df = pd.concat([raw_header_df, df.iloc[0:]]).reset_index(drop=True)
    return df

def drop_col_with_rework_flag(df):
    drop_col_name_ls = []
    for col_name in df.columns.values.tolist():
        if 'rework_flag' in col_name:
            drop_col_name_ls.append(col_name)
    df = df.drop(drop_col_name_ls, axis = 1)
    return df

def save_obj(file_name, obj):
    # .pkl
    with open(file_name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)

if __name__ == '__main__':
    # Parameter---------------------------------
    file_path = 'D:\\UserData\\Hsuenmei\\7B_AGM\\py\\data\\'
    read_file_name = 'L7B_AGM_ARY_CELL_His_temp.csv'
    output_file_name = 'L7B_AGM_ARY_CELL_His_temp.csv'

    # Main--------------------------------------
    df = pd.read_csv(file_path+read_file_name)
    print(df.shape)
    df = remove_col_with_same_value(df)
    df = remove_col_with_same_cate(df)
    df = replace_missing_value_by_column_mode(df)
    df = remove_row_duplicate(df)
    # df.to_csv(file_path+output_file_name, index=False)
    print(df.shape)