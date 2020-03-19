import os, json, requests
import numpy as np
import pandas as pd
import datetime
import plotly.express as px
from plotly.offline import init_notebook_mode 
init_notebook_mode(connected = True)
import plotly.io as pio
pio.templates.default = "simple_white"
import uuid

def get_hourly_data_for_azure(file_path, 
                              asset_id, 
                              asset_name, 
                              variable_name,  
                              measurement_type,
                              datestring_column_name = "DateString",
                              sensor_type = 'pressure'):
    
    df = pd.read_csv(file_path, parse_dates= True)
    df['sensor_info'] = variable_name
    df['date'] = pd.to_datetime(df['DateString']) # , format='%Y%m%d'
    df['Date_Hour'] = df['date'].dt.floor('H')
    g = df.groupby(["Date_Hour", "Unit"])
    df_hourly = g.agg({'AvgValue' : 'mean', 'MinValue' : 'min', 'MaxValue': 'max'}).reset_index()
    
    df_hourly = df_hourly.assign(asset_id = asset_id)
    df_hourly['asset_name'] = asset_name
    df_hourly['sensor_type']  = sensor_type
    df_hourly['measurement_type']  = measurement_type
    df_hourly['sensor_info'] = variable_name 
    
    df_hourly = df_hourly.rename(columns = {'Date_Hour': 'datetime', 'Unit': 'feature_unit'})
    df_hourly = pd.melt(df_hourly, 
                        id_vars=["asset_id","asset_name","sensor_type","measurement_type","feature_unit","sensor_info","datetime"],
                        value_vars = ["MinValue", "AvgValue", "MaxValue"], value_name="feature_value", var_name="feature_name")
    
    cols = [ 'asset_id', 'asset_name', 'sensor_type', 'measurement_type', 'feature_name', 'feature_value', 'feature_unit', 
            'sensor_info', 'datetime']
    df_hourly = df_hourly[cols]
    #df_hourly['uuid'] = [uuid.uuid4() for _ in range(len(df_hourly.index))]    
    df_hourly.sort_values(by='datetime').reset_index(drop=True, inplace=True)
    return(df_hourly)


def show_baselining_plot_azure(df_hourly_azure, duration = "all", 
                               asset_name = "Pump4", variable_name = "OutletPressure",
                               plot_type = "scatter", variable_value_col = "feature_value", 
                               feature_name = "AvgValue"):
    
    if (duration != "all"):
        df_hourly_azure = df_hourly_azure[df_hourly_azure['week'].isin(duration)]
        duration = ', '.join(map(str, duration))
    df_hourly_azure = df_hourly_azure[df_hourly_azure['feature_name'] == feature_name]
    df_hourly_azure = df_hourly_azure[df_hourly_azure['sensor_info'] == variable_name]
    #print(df_hourly_azure.head(2))
    #return(df_hourly_azure)
    
    title = variable_name + " " + asset_name + " for " + duration + " weeks "
    if (plot_type == "scatter"):
        fig = px.scatter(df_hourly_azure, x="week", y=variable_value_col, title=title)
    if (plot_type == "box"):
        fig = px.box(df_hourly_azure, x="week", y=variable_value_col, title=title)
    fig.update_layout(xaxis_type='category')
    fig.show()

    
def get_baselining_variables_azure(df, variable_name, asset_id, lst_filter_values,
                                   filter_column_name = "week", select_column_name = "feature_value",
                                   filter_metric = "AvgValue" ):
    """
    return mean and standard deviation for `select_column_name`
    """
    #x_var = df[df[filter_column_name].isin(lst_filter_values)][select_column_name]
    df = df[df[filter_column_name].isin(lst_filter_values)] # filtering weekly data
    df = df[df['sensor_info'] == variable_name]
    df = df[df['asset_id'] == asset_id]
    x_var = df[df['feature_name'] == filter_metric][select_column_name]
    return(np.mean(x_var), np.std(x_var))   




def convert_hourly_data_to_json(df_azure):
    df = df_azure.copy()
    df.loc[:, 'datetime'] = df.datetime.apply(str)
    #json_data = df.to_json(orient = 'records', default_handler=str, date_format = str )
    json_object = json.loads(df.to_json(orient = 'records', default_handler=str, date_format = str ))
    return json_object

def convert_baseline_data_to_json(df_azure):
    df = df_azure.copy()
    df.loc[:, 'baseline_datetime'] = df.baseline_datetime.apply(str)
    json_object = json.loads(df.to_json(orient = 'records', default_handler=str, date_format = str ))
    return(json_object)

def save_to_api(json_object, api):
    header = {"Content-Type": "application/json"} 
    res = requests.post(api, json=json_object, headers = header)
    return(res.status_code, res.reason)

def get_data_from_api(api_url = "https://hpp-cm-npd-aps-001.azurewebsites.net/AssetHourlyData/P4-xyzwab"):
    x = requests.get(api_url)
    return x # x.text is the data which can be loaded to a python dataframe by call pd.read_json.



def get_row_dict_baseline(asset_id, 
                          asset_name,
                          sensor_type,
                          measurement_type,
                          feature_name,
                          metric_name,
                          metric_value, 
                          feature_unit, 
                          sensor_info, 
                          baseline_datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
                          baseline_validity_month = 3, 
                          notes = None):
    return {'asset_id': asset_id,
            'asset_name': asset_name,
            'sensor_type': sensor_type,
            'measurement_type': measurement_type,
            'feature_name': 'AvgValue',
            'metric_name': metric_name,
            'metric_value': metric_value, 
            'feature_unit': feature_unit,
            'sensor_info': sensor_info, 
            'baseline_datetime': baseline_datetime,
            'baseline_validity_month' : baseline_validity_month, 
            'notes': notes}

def get_number_of_weeks(df_data, variable_name, asset_id):
    df_filt = df_data[(df_data['asset_id'] == asset_id) & (df_data['sensor_info'] == variable_name)]
    num_weeks = len(df_filt['week'].unique())
    print("Number of weeks for {} found:{}".format(variable_name, num_weeks ))
    
    
def add_date_components(df, date_col_name = 'DateTime'):
    df['d'] = df[date_col_name].dt.day
    df['dayofweek'] = df[date_col_name].dt.dayofweek
    df['year'] = df[date_col_name].dt.year
    df['month'] = df[date_col_name].dt.month
    df['hour'] = df[date_col_name].dt.hour
    df['minute'] = df[date_col_name].dt.minute
    df['second'] = df[date_col_name].dt.second
    df['week'] = df[date_col_name].dt.week
    return(df)

def get_file_path(data_folder, variable_name):
    return data_folder + variable_name + ".csv"

# Dummy function to get asset_id by asset_name as this is currently not available in data file
def fn_get_asset_id(asset_name):
    return('aid-' + asset_name)

def fn_drop_outliers(df_hourly, feature_name = "AvgValue", variable_value_col = "feature_value"):
    df_ = df_hourly.copy(deep=False)
    q1, q3 = df_.loc[df_['feature_name']==feature_name, variable_value_col].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower_lim = q1 - (iqr * 1.5)
    upper_lim = q3 + (iqr * 1.5)
    
    index_drop = df_[(((df_['feature_name']==feature_name) & (df_[variable_value_col] < lower_lim)) | 
                      ((df_['feature_name']==feature_name) & (df_[variable_value_col] > upper_lim)))].index
    df_.drop(index_drop, inplace=True)
    df_.reset_index(drop=True, inplace = True)
    
    return df_ 


def fn_remove_negative_values(df_, variable_value_col = "feature_value"):
    df_ = df_[(df_[variable_value_col] > 0)].reset_index(drop=True) #.copy(deep=False)
    return df_


#---------------------------------------------SHOULD BE DELETED BELOW THIS ----------------------------------------------------------

# def fn_drop_outliers(df_hourly, metric_name = "AvgValue", variable_value_col = "metric_value", 
#                      left_limit = 0.05, right_limit = 0.95):
#     df_ = df_hourly.copy(deep=False)
#     percent_05 , percent_95 = df_.loc[df_['metric_name']==metric_name,variable_value_col].quantile([left_limit, right_limit])
#     index_drop = df_[(((df_['metric_name']==metric_name) & (df_[variable_value_col] < percent_05)) | 
#                       ((df_['metric_name']==metric_name) & (df_[variable_value_col] > percent_95)))].index
#     df_.drop(index_drop, inplace=True)
#     df_.reset_index(drop=True, inplace = True)