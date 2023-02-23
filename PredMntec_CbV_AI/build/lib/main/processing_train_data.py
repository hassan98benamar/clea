import pandas as pd
import numpy as np
import os
import pickle
import csv
import time
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import PCA
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, mean_squared_error
from sklearn.metrics import mean_squared_error, mean_absolute_error
from datetime import datetime
from pymongo import MongoClient
from sklearn.linear_model import LinearRegression
from PredMntec_CbV_AI.main.pre_processing import PreProcess
from statsmodels.tsa.seasonal import seasonal_decompose

class ProcessTrain():

    pre = PreProcess()
    
    def read_data(self, ctrl_name=None):
        usecase = self.pre.usecase

        if self.pre.useDB and ctrl_name is not None:
            print('in read data mongodb condi')
            client = MongoClient('localhost', 27017)
            db = client['cbv']
            control_collection = db['Controls']
            if ctrl_name is None:
                cursor = control_collection.find()
            else:
                cursor = control_collection.find({"Control": ctrl_name})
            data = pd.DataFrame(list(cursor))

        elif ctrl_name is not None:
            path = os.path.join(self.pre.FILES_PATH, 'df_'+usecase+'.pkl')
            date_point_df = open(path, 'rb')
            data = pickle.load(date_point_df)
            date_point_df.close()

            data['Status'] = np.where(data['User Status']!= '', data['User Status'], data['Ai Status'])
            data = data.drop(['Ai Status', 'User Status'], axis = 1)
        
        else:
            path = os.path.join(self.pre.FILES_PATH, 'df_'+usecase+'.pkl')
            date_point_df = open(path, 'rb')
            data = pickle.load(date_point_df)
            date_point_df.close()

            data['Status'] = np.where(data['User Status']!= '', data['User Status'], data['Ai Status'])
            data = data.drop(['Ai Status', 'User Status'], axis = 1)

        return data

    # def read_data(self):
    #     usecase = self.pre.usecase

    #     if self.pre.useDB:
    #         client = MongoClient('localhost', 27017)
    #         db = client['cbv']
    #         control_collection = db['Controls']
    #         if ctrl_name is None:
    #             cursor = control_collection.find()
    #         else:
    #             cursor = control_collection.find({"Control": ctrl_name})
    #         data = pd.DataFrame(list(cursor))

    #     else:
    #         path = os.path.join(self.pre.FILES_PATH, 'df_'+usecase+'.pkl')
    #         date_point_df = open(path, 'rb')
    #         data = pickle.load(date_point_df)
    #         date_point_df.close()

    #         data['Status'] = np.where(data['User Status']!= '', data['User Status'], data['Ai Status'])
    #         data = data.drop(['Ai Status', 'User Status'], axis = 1)

    #     return data

    def days_diff(self, result_df):
        start_date = datetime.strptime(str(result_df["Date"].iloc[0]), "%Y-%m-%d")
        end_date = datetime.strptime(str(result_df["Date"].iloc[-1]), "%Y-%m-%d")
        diff = end_date - start_date
        return start_date, end_date, diff.days
    
    def add_missing_dates(self, result_df):
        #Adding -1 to dates not present in dataset - Mntenc = False
        idx = pd.period_range(result_df["Date"].iloc[0], result_df["Date"].iloc[-1])
        idx = [x for x in idx if x.weekday != 6]
        add_dict = dict()
        for dt in idx:
            if str(dt) not in list(result_df['Date']):
                add_dict[str(dt)] = -1
        add_df = pd.DataFrame.from_dict(add_dict.items())
        add_df.columns = ['Date', 'Status Code']
        return add_df

    def add_values(self, add_df, result_df):
        #Merging both the dfs, original and gap dates
        final_df = pd.DataFrame()
        final_df = pd.concat([result_df, add_df])
        final_df.sort_values(["Date", "Status Code"], inplace = True)
        final_df = final_df.drop_duplicates(subset=['Date'], keep='last')
        # final_df.sort_values("Status Code", inplace = True)
        final_df = final_df.reset_index()
        final_df = final_df.drop(['index'], axis=1)
        return final_df

    def add_back_dates_and_values(self, start_date, diff_days, final_df, m=2):
        #Adding back dates
        back_start_date = start_date - pd.DateOffset(days=m*diff_days+m)
        back_end_date = start_date - pd.DateOffset(days=1)
        back_idx = pd.period_range(back_start_date, back_end_date)
        back_idx = [x for x in back_idx if x.weekday != 6]
        back_idx = back_idx[-len(final_df)*m:]
        # print(back_idx)
        #Adding back values
        back_df = pd.DataFrame()
        back_df['Date'] = back_idx
        # print(len(back_df['Date']))
        back_df['Date'] = back_df['Date'].astype(str)

        st_code = final_df['Status Code'].to_list()
        # print(len(st_code))
        back_df['Status Code'] = st_code + st_code
        # back_df['Status'] = final_df['Status'] + final_df['Status']
        # back_df['Status'] = final_df['Status'] + final_df['Status']

        #Merging back and current df
        final_df = pd.concat([final_df, back_df])
        final_df.sort_values("Date", inplace = True)
        # final_df[final_df['Status Code'] == 2]
        final_df = final_df.reset_index()
        final_df = final_df.drop(['index'], axis=1)
        return(final_df)

    def windowed_dataset_numpy(self, final_df):
        # print(final_df)
        window_size = 60
        time_col = final_df['Date'].values[window_size:]
        window_size = window_size + 1
        temps = final_df['Status Code']

        data_df = pd.concat([temps] + [temps.shift(-x) for x in range(1, window_size)], axis=1)
        data_df.columns = ['t' + str(x) for x in range(-window_size+1, 1)]
        data_df.dropna(inplace=True)
        data_df['Date'] = time_col
        return data_df

    def feature_eng(self, df):
        # print("before", df.head())
        dim_reduce_bool=True
    #     scale=self.model_params['SCALE']
        n_components = 10
        drop_cols = ['t0', 'Date']
        trans_cols = df.columns.difference(drop_cols)
        req_df = df[trans_cols]

    #     if scale.lower().strip() == 'pre':
        scaler = Normalizer()
        req_df = pd.DataFrame(scaler.fit_transform(req_df))

        if dim_reduce_bool:
            # print("in dim_reduce", req_df.head())
            dim_reduce = PCA(n_components=n_components)
            req_df = pd.DataFrame(dim_reduce.fit_transform(req_df))
            # print("after", req_df.head())

    #     if scale.lower().strip() == 'post':
    #         scaler = Normalizer()
    #         req_df = pd.DataFrame(scaler.fit_transform(req_df))

        req_df[drop_cols] = df[drop_cols]
        return req_df

    def train_test_split(self, req_df):
        split_data = 0.8
        split_time = int(req_df.shape[0] * split_data)
        
        x_train = req_df.drop(['Date', 't0'], axis=1).values[:split_time]
        y_train = req_df['t0'].values[:split_time]

        x_test = req_df.drop(['Date', 't0'], axis=1).values[split_time:]
        y_test = req_df['t0'].values[split_time:]
        
        return x_train, y_train, x_test, y_test
    
    def build_model(self, x_train, y_train, x_test, y_test):
        model = xgb.XGBClassifier(objective='multi:softprob', n_estimators=100, learning_rate=0.3, num_class=len(set(y_train)), eval_metric='mlogloss')
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        acc = round(accuracy_score(y_test, y_pred),2)
        recall = (recall_score(y_test, y_pred, average='micro'))
        rms = mean_squared_error(y_test, y_pred, squared=False)
        return model, acc, recall, rms

    def data_to_consider(self, window_value_days, global_df):
        considerable_size = (window_value_days + 1) * 24 
        pred_value_dep_data_df = global_df[-considerable_size:]
        return pred_value_dep_data_df

    def data_to_consider_daily(self, window_value_days, global_df):
        considerable_size = (window_value_days + 1)
        pred_value_dep_data_df = global_df[-considerable_size:]
        return pred_value_dep_data_df

    def pre_process_data_hourly(self, global_df):
        global_df.sort_values("Date", inplace=True)
        global_df['Hour'] = global_df['Timestamp'].apply(lambda x: datetime.strptime(x, '%Y-%m-%dT%H:%M:%S.%fZ')).dt.hour
        global_df = global_df.reset_index(drop=True)
        global_df = global_df.drop(columns='Timestamp')

        # print(global_df)

        if 'GlobalStatus' in global_df.columns:
            # global_df['GlobalStatus'] = global_df['GlobalStatus'].replace(['NOT FOUND'], 'NOT_FOUND')
            # global_df['GlobalStatus'] = global_df['GlobalStatus'].map(lambda x: x.items())
            global_df = pd.DataFrame(global_df.groupby([global_df['Date'], global_df['Hour'], 'GlobalStatus'])['GlobalStatus'].count()).rename(columns = {'GlobalStatus':'Count'})
            global_df.reset_index()
            global_df = global_df.pivot_table(index=['Date', 'Hour'], columns=['GlobalStatus'], values = 'Count').reset_index().fillna(0)
        if 'Status' in global_df.columns:
            # global_df['Status'] = global_df['Status'].replace(['NOT FOUND'], 'NOT_FOUND')
            # global_df['Status'] = global_df['Status'].map(lambda x: x.item())
            global_df = pd.DataFrame(global_df.groupby([global_df['Date'], global_df['Hour'], 'Status'])['Status'].count()).rename(columns={'Status': 'Count'}).reset_index()
            global_df = global_df.pivot_table(index=['Date', 'Hour'], columns=['Status'], values='Count').reset_index().fillna(0)
        return global_df

    def pre_process_data_daily(self, global_df):
        global_df.sort_values("Date", inplace=True)
        global_df = global_df.reset_index(drop=True)
        global_df = global_df.drop(columns='Timestamp')

        # print(global_df)

        if 'GlobalStatus' in global_df.columns:
            # global_df['GlobalStatus'] = global_df['GlobalStatus'].replace(['NOT FOUND'], 'NOT_FOUND')
            # global_df['GlobalStatus'] = global_df['GlobalStatus'].map(lambda x: x.items())
            global_df = pd.DataFrame(global_df.groupby([global_df['Date'], 'GlobalStatus'])['GlobalStatus'].count()).rename(columns = {'GlobalStatus':'Count'})
            global_df.reset_index()
            global_df = global_df.pivot_table(index=['Date'], columns=['GlobalStatus'], values = 'Count').reset_index().fillna(0)
        if 'Status' in global_df.columns:
            # global_df['Status'] = global_df['Status'].replace(['NOT FOUND'], 'NOT_FOUND')
            # global_df['Status'] = global_df['Status'].map(lambda x: x.item())
            global_df = pd.DataFrame(global_df.groupby([global_df['Date'], 'Status'])['Status'].count()).rename(columns={'Status': 'Count'}).reset_index()
            global_df = global_df.pivot_table(index=['Date'], columns=['Status'], values='Count').reset_index().fillna(0)
        return global_df

    def add_missing_hours(self, df):

        # add_hours_df = pd.DataFrame(columns =['Date', 'Hour', 'FAIL', 'PASS'])
        # for d in sorted(set(df['Date'])):
        #     #     print(d)
        #     for i in self.pre.range_hours:
        #         if df[(df['Hour'] == i) & (df['Date'] == d)].empty:
        #             add_hours_df.loc[len(add_hours_df.index)] = [d, i, 0, 0]
        # return add_hours_df

        add_hours_df = pd.DataFrame(columns=df.columns)
        # print(df.columns)
        # print(df.head())
        for d in sorted(set(df['Date'])):
            for i in self.pre.range_hours:
                if df[(df['Hour'] == i) & (df['Date'] == d)].empty:
                    list_to_add = [d, i]
                    for _ in range(len(df.columns) - 2):
                        list_to_add.append(0)
                    add_hours_df.loc[len(add_hours_df.index)] = list_to_add
        return add_hours_df

    # def add_missing_dates_hourly(self, df):
    #     idx = pd.period_range(df["Date"].iloc[0], df["Date"].iloc[-1])
    #     idx = [x for x in idx if x.weekday != 6]
    #     add_dates_df = pd.DataFrame(columns =['Date', 'Hour', 'FAIL', 'PASS'])
    #     for dt in idx:
    #         if str(dt) not in list(df['Date']):
    #             for i in self.pre.range_hours:
    #                 add_dates_df.loc[len(add_dates_df.index)] = [str(dt), i, 0, 0]
    #     return add_dates_df

    def add_missing_dates_hourly(self, df):

        idx = pd.period_range(df["Date"].iloc[0], df["Date"].iloc[-1])
        idx = [x for x in idx if x.weekday != 6]
        add_dates_df = pd.DataFrame(columns=df.columns)
        for d in idx:
            if str(d) not in list(df['Date']):
                for i in self.pre.range_hours:
                    list_to_add = [d, i]
                    for _ in range(len(df.columns) - 2):
                        list_to_add.append(0)
                    add_dates_df.loc[len(add_dates_df.index)] = list_to_add
        return add_dates_df

    def add_missing_hours_and_dates(self, df, status_cols):
        idx = pd.period_range(df["Date"].iloc[0], df["Date"].iloc[-1])
        idx = [x for x in idx if x.weekday != 6]
    #     date_list = list(set(map(str, idx)) - set(df["Date"]))
        date_list = list(map(str, idx))
        hour_list = list(self.pre.range_hours) * len(date_list)
        date_list = date_list * len(list(self.pre.range_hours))
        date_list = sorted(date_list)
        s = []
        # for i in range(len(status_cols)):
        #     s.append([0] * len(hour_list))
        #Adding MEDIAN for missing days n hours
        for i in status_cols:
            s.append([df[i].median()] * len(hour_list))
        l_to_zip = [date_list, hour_list] + s
        l = list(zip(*l_to_zip))
        add_dates_df = pd.DataFrame(l, columns=df.columns)
        return add_dates_df

    # Filling Missing Dates with Null

    def add_missing_dates_and_hours_using_null(self, df, status_cols):
        idx = pd.period_range(df["Date"].iloc[0], df["Date"].iloc[-1])
        idx = [x for x in idx if x.weekday != 6]

        date_list = list(map(str, idx))
        hour_list = list(self.pre.range_hours) * len(date_list)
        date_list = date_list * len(list(self.pre.range_hours))
        date_list = sorted(date_list)
    
        s = []
    
        for i in status_cols:
            s.append([np.nan] * len(hour_list))
        
        l_to_zip = [date_list, hour_list] + s
        l = list(zip(*l_to_zip))
        add_dates_df = pd.DataFrame(l, columns = df.columns)
    
        return add_dates_df

    # Filling Missing Dates with Null for Daily
    def add_missing_dates_without_hours_using_null(self, df, status_cols):
    
        idx = pd.period_range(df["Date"].iloc[0], df["Date"].iloc[-1])
        idx = [x for x in idx if x.weekday != 6]
        
        date_list = list(map(str, idx))
        date_list = sorted(date_list)
        
        date_list_start = date_list[0]
        date_list_end = date_list[-1]

        dates = pd.date_range(date_list_start, date_list_end)
        new_df = pd.DataFrame(dates, columns=['Date'])
        
        for i in status_cols :        
            new_df[i] = np.nan
            
        new_df['Date'] = new_df['Date'].dt.strftime('%Y-%m-%d')

        pre_final_df = pd.concat([df, new_df])
        pre_final_df = pre_final_df.sort_values(['Date']).drop_duplicates(subset=['Date'], keep='first').reset_index(drop=True)
        
        return pre_final_df

    # Filling Null Values Using LR Model    
    def fill_nan_values_using_LR(self, df, status_cols):
        
        df_fill_null = df[df['PASS'].isnull() == True]
        
        for i in status_cols:
            
            lr = LinearRegression()
        
            testdf = df[df[i].isnull() == True] 
            # print(testdf.shape)
            traindf = df[df[i].isnull() == False]
            
            y = traindf[i]
            
            date_col = ['Date']
            
            traindf.drop(status_cols + date_col, axis = 1, inplace = True)
            lr.fit(traindf, y)
            
            testdf.drop(status_cols + date_col, axis = 1, inplace = True)
            pred = lr.predict(testdf)
            testdf[i]= pred
            
            df_fill_null[i] = testdf[i]
        
        return df_fill_null

    # Filling Null Values Using LR Model for Daily
    def fill_nan_values_using_LR_Daily(self, df, status_cols):     
            
        temp_df = df.copy()
        temp_df['Day'] = temp_df['Date'].dt.day
        df_fill_null = temp_df[temp_df['PASS'].isnull() == True]
        
        for i in status_cols:
            
            lr = LinearRegression()
        
            testdf = temp_df[temp_df[i].isnull() == True]        
            traindf = temp_df[temp_df[i].isnull() == False]
            
            y = traindf[i]
            
            date_col = ['Date']
            
            traindf.drop(status_cols + date_col, axis = 1, inplace = True)
            lr.fit(traindf, y)
            
            testdf.drop(status_cols + date_col, axis = 1, inplace = True)
            pred = lr.predict(testdf)
            testdf[i]= pred
            
            df_fill_null[i] = testdf[i]
        
        df_fill_null.drop(columns=['Day'], inplace=True)
        return df_fill_null


    def seasonality_removal(self, df, cols, days_in_week):
    
        updated_df = df 
            
        days_in_week = days_in_week * 24
            
        for lbl in cols:
                
            lbl_mean = df[lbl].mean()            
            lbl_std = df[lbl].std()            
                
            if (lbl_mean > 1 or lbl_mean < -1) and (lbl_std > 2 or lbl_std < -2):
                
                for i in range(0, len(df[lbl])) :
                    if df[lbl].iloc[i] == 0 :
                        df[lbl].iloc[i] += 0.1
                            
                count = (df[lbl] == 0).sum()
                
                model_to_be_used = ""
                
                if count > 1 :
                    model_to_be_used = "additive"
                else :
                    model_to_be_used = "multiplicative"

                decompose_result = seasonal_decompose(df[lbl], model = model_to_be_used, period = days_in_week)

                trend = decompose_result.trend
                seasonal = decompose_result.seasonal
                residual = decompose_result.resid
                
                for i in range(0, len(updated_df[lbl])) :
                    
                    if seasonal[i] > 0 :
                        updated_df[lbl].iloc[i] = updated_df[lbl].iloc[i] - seasonal[i]
                    else :
                        updated_df[lbl].iloc[i] = updated_df[lbl].iloc[i] - (df[lbl].median() * 0.75)
                        
        return updated_df


    def seasonality_removal_for_daily(self, df, cols, days_in_week):

        updated_df = df.copy()
            
        days_in_week = days_in_week
            
        for lbl in cols:
                
            lbl_mean = df[lbl].mean()            
            lbl_std = df[lbl].std()            
                
            if (lbl_mean > 1 or lbl_mean < -1) and (lbl_std > 2 or lbl_std < -2):
                
                for i in range(0, len(df[lbl])) :
                    if df[lbl].iloc[i] == 0 :
                        df[lbl].iloc[i] += 0.1
                            
                count = (df[lbl] == 0).sum()
                
                model_to_be_used = ""
                
                if count > 1 :
                    model_to_be_used = "additive"
                else :
                    model_to_be_used = "multiplicative"

                decompose_result = seasonal_decompose(df[lbl], model = model_to_be_used, period = days_in_week)

                trend = decompose_result.trend
                seasonal = decompose_result.seasonal
                residual = decompose_result.resid
                
                for i in range(0, len(updated_df[lbl])) :
                    
                    if seasonal[i] > 0 :
                        updated_df[lbl].iloc[i] = updated_df[lbl].iloc[i] - seasonal[i]
                    else :
                        updated_df[lbl].iloc[i] = updated_df[lbl].iloc[i] - (df[lbl].median() * 0.75)
                        
        return updated_df


    def windowed_dataset_numpy_hourly(self, df, cols):
        # window_size = 60

        window_size = 30

        date_col = df['Date'].values[window_size:]
        hour_col = df['Hour'].values[window_size:]
        
        window_size = window_size + 1
        temps = df[cols]
        data_df = pd.concat([temps] + [temps.shift(-x) for x in range(1, window_size)], axis=1)
        
        data_df.columns = ['t' + str(x) for x in range(len(cols)*(-window_size)+1, 1)]
        data_df.dropna(inplace=True)
        
        data_df['Date'] = date_col
        data_df['Hour'] = hour_col
        return data_df

    def windowed_dataset_numpy_daily(self, df, cols):
        window_size = 30

        date_col = df['Date'].values[window_size:]
        
        window_size = window_size + 1
        temps = df[cols]
        data_df = pd.concat([temps] + [temps.shift(-x) for x in range(1, window_size)], axis=1)
        
        data_df.columns = ['t' + str(x) for x in range(len(cols)*(-window_size)+1, 1)]
        data_df.dropna(inplace=True)
        
        data_df['Date'] = date_col

        return data_df

    def feature_eng_hourly(self, df, cols):

        dim_reduce_bool = True
        n_components = len(cols) * 5
        
        t_cols = ['t' + str(x) for x in range(-len(cols)+1, 1)]
        h_d_cols = ['Date', 'Hour']
        drop_cols = t_cols + h_d_cols
        trans_cols = df.columns.difference(drop_cols)
        req_df = df[trans_cols]

        scaler = Normalizer()
        req_df = pd.DataFrame(scaler.fit_transform(req_df))

        if dim_reduce_bool:
            
            dim_reduce = PCA(n_components=n_components)
            req_df = pd.DataFrame(dim_reduce.fit_transform(req_df))            

        req_df[drop_cols] = df[drop_cols]
        return req_df

    def feature_eng_daily(self, df, cols):
        
        dim_reduce_bool = True
        n_components = len(cols) * 3
        
        t_cols = ['t' + str(x) for x in range(-len(cols)+1, 1)]
        
        h_d_cols = ['Date']
        drop_cols = t_cols + h_d_cols
        trans_cols = df.columns.difference(drop_cols)
        req_df = df[trans_cols]

        scaler = Normalizer()
        req_df = pd.DataFrame(scaler.fit_transform(req_df))

        if dim_reduce_bool:
            
            dim_reduce = PCA(n_components=n_components)
            req_df = pd.DataFrame(dim_reduce.fit_transform(req_df))

        req_df[drop_cols] = df[drop_cols]
        return req_df

    # Feature Scaling & Engineerng On Train & Test Data
    def feature_engg_train_test_data_daily(self, df_x_train, df_y_train, df_x_test, df_y_test, cols):
    
        dim_reduce_bool = True
        n_components = len(cols) * 3
        print(n_components)
        
        req_df_x_train = df_x_train
        req_df_x_test = df_x_test
        
        req_df_y_train = df_y_train
        req_df_y_test = df_y_test

        scaler = Normalizer()
        
        scaler.fit(req_df_x_train)
        
        req_df_x_train = pd.DataFrame(scaler.transform(req_df_x_train))
        req_df_x_test = pd.DataFrame(scaler.transform(req_df_x_test))
        
        if dim_reduce_bool:
            
            dim_reduce = PCA(n_components = n_components)
            
            dim_reduce.fit(req_df_x_train)
            
            req_df_x_train = pd.DataFrame(dim_reduce.transform(req_df_x_train))
            req_df_x_test = pd.DataFrame(dim_reduce.transform(req_df_x_test))
            
            x_train = req_df_x_train.values
            x_test = req_df_x_test.values
            y_train = req_df_y_train.values
            y_test = req_df_y_test.values

            print(x_train.shape)
        return x_train, y_train, x_test, y_test


    def feature_engg_train_test_data_hourly(self, df_x_train, df_y_train, df_x_test, df_y_test, cols):
    
        dim_reduce_bool = True
        n_components = len(cols) * 5
        print(n_components)
        
        req_df_x_train = df_x_train
        req_df_x_test = df_x_test
        
        req_df_y_train = df_y_train
        req_df_y_test = df_y_test

        scaler = Normalizer()
        
        scaler.fit(req_df_x_train)
        
        req_df_x_train = pd.DataFrame(scaler.transform(req_df_x_train))
        req_df_x_test = pd.DataFrame(scaler.transform(req_df_x_test))
        
        if dim_reduce_bool:
            
            dim_reduce = PCA(n_components = n_components)
            
            dim_reduce.fit(req_df_x_train)
            
            req_df_x_train = pd.DataFrame(dim_reduce.transform(req_df_x_train))
            req_df_x_test = pd.DataFrame(dim_reduce.transform(req_df_x_test))
            
            x_train = req_df_x_train.values
            x_test = req_df_x_test.values
            y_train = req_df_y_train.values
            y_test = req_df_y_test.values

            print(x_train.shape)
        return x_train, y_train, x_test, y_test

    
    def train_test_split_hourly(self, req_df, cols):
        split_data = 0.8
        split_time = int(req_df.shape[0] * split_data)
        t_cols = ['t' + str(x) for x in range(-len(cols)+1, 1)]
        
        h_d_cols = ['Date', 'Hour']
        drop_cols = t_cols + h_d_cols
        
        x_train = req_df.drop(drop_cols, axis=1).values[:split_time]
        y_train = req_df[t_cols].values[:split_time]

        x_test = req_df.drop(drop_cols, axis=1).values[split_time:]
        y_test = req_df[t_cols].values[split_time:]

        return x_train, y_train, x_test, y_test

    def train_test_split_before_pca(self, req_df, cols):
        split_data = 0.8
        split_time = int(req_df.shape[0] * split_data)
        t_cols = ['t' + str(x) for x in range(-len(cols)+1,1)]
        
        print(t_cols)
        
        h_d_cols = ['Date', 'Hour']
        drop_cols = t_cols + h_d_cols
        
        x_train = req_df.drop(drop_cols, axis=1)
        x_train = x_train[:split_time]
        
        y_train = req_df[t_cols]
        y_train = y_train[:split_time]

        x_test = req_df.drop(drop_cols, axis=1)
        x_test = x_test[split_time:]
        
        y_test = req_df[t_cols]
        y_test = y_test[split_time:]
        
        return x_train, y_train, x_test, y_test


    def train_test_split_before_pca_daily(self, req_df, cols):
        split_data = 0.8
        split_time = int(req_df.shape[0] * split_data)
        t_cols = ['t' + str(x) for x in range(-len(cols)+1,1)]

        print(t_cols)

        h_d_cols = ['Date']
        drop_cols = t_cols + h_d_cols

        x_train = req_df.drop(drop_cols, axis=1)
        x_train = x_train[:split_time]

        y_train = req_df[t_cols]
        y_train = y_train[:split_time]

        x_test = req_df.drop(drop_cols, axis=1)
        x_test = x_test[split_time:]

        y_test = req_df[t_cols]
        y_test = y_test[split_time:]

        return x_train, y_train, x_test, y_test



    def build_global_model(self, x_train, y_train, x_test, y_test, cols):
        # model = MultiOutputRegressor(xgb.XGBRegressor(n_estimators=50, learning_rate=0.5, alpha=0.5, random_state=0)).fit(x_train, y_train)

        model = MultiOutputRegressor(xgb.XGBRegressor(n_estimators = 150, learning_rate = 0.5, alpha = 0.5, random_state = 0)).fit(x_train, y_train)

        y_pred = model.predict(x_test)
        y_pred = np.around(y_pred, decimals=0)
        
        mse = []
        mae = []
        for i in range(len(cols)):
            mse.append(mean_squared_error(y_test[:, i], y_pred[:, i]))
            mae.append(mean_absolute_error(y_test[:, i], y_pred[:, i]))

        return model, mse, mae

    def pre_process_global_data_daily(self, global_df, out_status):
        print(global_df)
        global_df[out_status] = global_df[out_status].replace(['NOT FOUND'], 'NOT_FOUND')
        global_df.sort_values("Date", inplace=True)
        global_df = global_df.drop(columns='Timestamp').reset_index(drop=True)
        if not isinstance(global_df[out_status].iloc[0], str):
            global_df[out_status] = global_df[out_status].map(lambda x: x.item())
        global_df = pd.DataFrame(global_df.groupby([global_df['Date'], out_status])[out_status].count()).rename(columns={out_status: 'Count'}).reset_index()
        global_df = global_df.pivot_table(index=['Date'], columns=[out_status], values='Count').reset_index().fillna(0)
        return global_df

    def add_missing_dates_without_hours(self, df, max_date):
        idx = pd.period_range(df["Date"].iloc[0], max_date)
        idx = [x for x in idx if x.weekday != 6]
        add_dates_df = pd.DataFrame(columns=df.columns)

        #Adding MEDIAN for missing days n hours
        status_cols = df.columns.drop('Date')
        s = []
        for i in status_cols:
            s.append(df[i].median())

        for dt in idx:
            if str(dt) not in list(df['Date']):
                add_dates_df.loc[len(add_dates_df.index)] = [str(dt)] + s
        return add_dates_df

    def add_back_dates_global_daily(self, final_df, status_cols):
        #Adding back dates
        # print(final_df)
        m = 1
        start_date = datetime.strptime(str(final_df["Date"].iloc[0]), "%Y-%m-%d")
        end_date = datetime.strptime(str(final_df["Date"].iloc[-1]), "%Y-%m-%d")
        # print(start_date, end_date)
        diff = end_date - start_date
        back_start_date = start_date - pd.DateOffset(days=m*diff.days+m)
        back_end_date = start_date - pd.DateOffset(days=1)
        back_idx = pd.period_range(back_start_date, back_end_date)
        back_idx = [x for x in back_idx if x.weekday != 6]
        back_idx = back_idx[-len(final_df)*m:]
        #Adding back values
        back_df = pd.DataFrame()
        back_df['Date'] = back_idx
        # print(len(back_df['Date']))
        back_df['Date'] = back_df['Date'].astype(str)

        back_df = back_df.join(final_df[status_cols])

        # pass_list = final_df['PASS'].to_list()
        # # print(len(pass_list))
        # back_df['PASS'] = m * pass_list
        # pass_list = final_df['FAIL'].to_list()
        # back_df['FAIL'] = m*pass_list
        return back_df


    def windowed_dataset_numpy_global_daily(self, df, cols):
        window_size = 60
        time_col = df['Date'].values[window_size:]
        window_size = window_size + 1
        # temps = df[['PASS', 'FAIL']]
        temps = df[cols]
        data_df = pd.concat([temps] + [temps.shift(-x) for x in range(1, window_size)], axis=1)
        # print(data_df)
        data_df.columns = ['t' + str(x) for x in range(len(cols)*(-window_size)+1, 1)]
        data_df.dropna(inplace=True)
        data_df['Date'] = time_col
        return data_df

    def feature_eng_global_daily(self, df, cols):
        # print("before", df.head())
        dim_reduce_bool = True
        n_components = 10
        # drop_cols = ['t0', 't-1', 'Date']
        t_cols = ['t' + str(x) for x in range(-len(cols)+1, 1)]
        # print("t_cols", t_cols)
        h_d_cols = ['Date']
        drop_cols = t_cols + h_d_cols
        trans_cols = df.columns.difference(drop_cols)
        req_df = df[trans_cols]

        scaler = Normalizer()
        req_df = pd.DataFrame(scaler.fit_transform(req_df))

        if dim_reduce_bool:
            # print("in dim_reduce", req_df.head())
            dim_reduce = PCA(n_components=n_components)
            req_df = pd.DataFrame(dim_reduce.fit_transform(req_df))
            # print("after", req_df.head())

        req_df[drop_cols] = df[drop_cols]
        # print(req_df)
        return req_df

    def train_test_split_global_daily(self, req_df, cols):
        split_data = 0.86
        split_time = int(req_df.shape[0] * split_data)
        t_cols = ['t' + str(x) for x in range(-len(cols)+1, 1)]
        # print(req_df.head(5))
        # print(cols)
        # print(t_cols)
        h_d_cols = ['Date']
        drop_cols = t_cols + h_d_cols
        # print(drop_cols)
        x_train = req_df.drop(drop_cols, axis=1).values[:split_time]
        y_train = req_df[t_cols].values[:split_time]

        x_test = req_df.drop(drop_cols, axis=1).values[split_time:]
        y_test = req_df[t_cols].values[split_time:]

        return x_train, y_train, x_test, y_test


if __name__ == '__main__':
    traindata = ProcessTrain()
    s_t = time.time()
    print(traindata.read_data())
    print(time.time() - s_t)
