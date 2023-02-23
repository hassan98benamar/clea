import pandas as pd
import time
import os
import pickle
from datetime import datetime

from multiprocessing import Pool
from PredMntec_CbV_AI.main.processing_train_data import ProcessTrain
from PredMntec_CbV_AI.main.save_model import Save
from PredMntec_CbV_AI.main.pre_processing import PreProcess
from itertools import repeat

import warnings
warnings.filterwarnings('ignore')


class Train:

    '''
    A class to train the model on the CbV data

    Methods
    =======

    '''

    pre = PreProcess()
    traindata = ProcessTrain()
    save = Save()

    back_date_counter = 2

    def process_data(self, result_df):
        # print(result_df.shape)
        result_df = result_df.drop(['Status', 'Timestamp'], axis=1)
        add_df = self.traindata.add_missing_dates(result_df)
        final_df = self.traindata.add_values(add_df, result_df)
        # print(final_df.shape)
        start_date, end_date, diff_days = self.traindata.days_diff(result_df)
        final_df = self.traindata.add_back_dates_and_values(start_date, diff_days, final_df, self.back_date_counter)
        # print(final_df)
        # exit(0)
        # print(set(final_df['Status']))
        data_df = self.traindata.windowed_dataset_numpy(final_df)
        return data_df

    # def process_control_data_hourly(self, result_df):
    #     add_hours_df = self.traindata.add_missing_hours(result_df)
    #     start_date, end_date, diff_days = self.traindata.days_diff(result_df)
    #     add_dates_df = self.traindata.add_missing_dates_hourly(result_df)
    #     final_df = pd.concat([result_df, add_hours_df, add_dates_df])
    #     final_df = final_df.sort_values(['Date', 'Hour']).reset_index(drop=True)
    #     final_df['FAIL'] = final_df['FAIL'].astype(int)
    #     final_df['PASS'] = final_df['PASS'].astype(int)
    #     final_df = self.traindata.windowed_dataset_numpy_hourly(final_df)
    #     return final_df

    def process_hourly_data(self, global_df):
        st = time.time()
        global_df = self.traindata.pre_process_data_hourly(global_df)
        
        status_cols = list(global_df.columns)
        status_cols.remove('Date')
        status_cols.remove('Hour')
        
        # add_df = self.traindata.add_missing_hours_and_dates(global_df, status_cols)

        # Adding NaN values in place of missing Values
        add_df = self.traindata.add_missing_dates_and_hours_using_null(global_df, status_cols)
        add_df_nan = pd.concat([global_df, add_df])

        add_df_nan['Date'] = add_df_nan['Date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
        add_df_nan = add_df_nan.sort_values(['Date','Hour']).drop_duplicates(subset=['Date', 'Hour'], keep='first').reset_index(drop=True)

        # Adding values in place of missing values using Linear Regression
        df_fill_null = self.traindata.fill_nan_values_using_LR(add_df_nan, status_cols)

        pre_final_df = pd.concat([add_df_nan, df_fill_null])        
        pre_final_df = pre_final_df.sort_values(['Date', 'Hour']).drop_duplicates(subset=['Date', 'Hour'], keep='last').reset_index(drop=True)

        # Removing Seasonality Component From the Original Data
        final_df = self.traindata.seasonality_removal(pre_final_df, status_cols, 4)     

        # Creating Windowed Dataset
        final_df = self.traindata.windowed_dataset_numpy_hourly(final_df, status_cols)
      
        return final_df, status_cols

    def process_daily_data(self, global_df):
        st = time.time()
        global_df = self.traindata.pre_process_data_daily(global_df)
        
        status_cols = list(global_df.columns)
        status_cols.remove('Date')
    
        # add_df = self.traindata.add_missing_hours_and_dates(global_df, status_cols)

        
        # Adding NaN values in place of missing Values
        add_df = self.traindata.add_missing_dates_without_hours_using_null(global_df, status_cols)
        add_df_nan = pd.concat([global_df, add_df])

        add_df_nan['Date'] = add_df_nan['Date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
        add_df_nan = add_df_nan.sort_values(['Date']).drop_duplicates(subset=['Date'], keep='first').reset_index(drop=True)
        
        # Adding values in place of missing values using Linear Regression
        df_fill_null = self.traindata.fill_nan_values_using_LR_Daily(add_df_nan, status_cols)

        pre_process_df = pd.concat([add_df_nan, df_fill_null])
        pre_final_df = pre_process_df[pre_process_df['PASS'].notna()]   
        pre_final_df = pre_final_df.sort_values(['Date']).drop_duplicates(subset=['Date'], keep='first').reset_index(drop=True)

        # Removing Seasonality Component From the Original Data
        final_df = self.traindata.seasonality_removal_for_daily(pre_final_df, status_cols, 4)     

        # Creating Windowed Dataset
        final_df = self.traindata.windowed_dataset_numpy_daily(final_df, status_cols)
      
        return final_df, status_cols


    def process_daily_data_for_prediction(self, global_df):
        st = time.time()
        global_df = self.traindata.pre_process_data_daily(global_df)
        
        status_cols = list(global_df.columns)
        status_cols.remove('Date')

        # Adding NaN values in place of missing Values
        add_df = self.traindata.add_missing_dates_without_hours_using_null(global_df, status_cols)
        add_df_nan = pd.concat([global_df, add_df])

        add_df_nan['Date'] = add_df_nan['Date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
        add_df_nan = add_df_nan.sort_values(['Date']).drop_duplicates(subset=['Date'], keep='first').reset_index(drop=True)

        # Adding values in place of missing values using Linear Regression
        df_fill_null = self.traindata.fill_nan_values_using_LR_Daily(add_df_nan, status_cols)

        pre_process_df = pd.concat([add_df_nan, df_fill_null])
        pre_final_df = pre_process_df[pre_process_df['PASS'].notna()]   
        pre_final_df = pre_final_df.sort_values(['Date']).drop_duplicates(subset=['Date'], keep='first').reset_index(drop=True)

        # Data to Consider on which the future value is dependent upon
        df_to_be_consider = self.traindata.data_to_consider_daily(60, pre_final_df)
        
        # Creating Windowed Dataset
        final_df = self.traindata.windowed_dataset_numpy_daily(df_to_be_consider, status_cols)
      
        return final_df, status_cols


    def process_hourly_data_for_prediction(self, global_df):
        st = time.time()
        global_df = self.traindata.pre_process_data_hourly(global_df)

        status_cols = list(global_df.columns)
        status_cols.remove('Date')
        status_cols.remove('Hour')

        # Adding NaN values in place of missing Values
        add_df = self.traindata.add_missing_dates_and_hours_using_null(global_df, status_cols)

        add_df_nan = pd.concat([global_df, add_df])
        # print(add_df_nan)
        
        add_df_nan['Date'] = add_df_nan['Date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
        add_df_nan = add_df_nan.sort_values(['Date','Hour']).drop_duplicates(subset=['Date', 'Hour'], keep = 'first').reset_index(drop=True)
        # print(add_df_nan)

        # Adding values in place of missing values using Linear Regression
        df_fill_null = self.traindata.fill_nan_values_using_LR(add_df_nan, status_cols)
        # print(df_fill_null)

        # Creating Final Dataframe
        pre_final_df = pd.concat([add_df_nan, df_fill_null])
        final_df = pre_final_df.sort_values(['Date','Hour']).drop_duplicates(subset=['Date', 'Hour'], keep = 'last').reset_index(drop=True)
        # print(final_df)

        # Data to Consider on which the future value is dependent upon
        df_to_be_consider = self.traindata.data_to_consider(30, final_df)
        # print(df_to_be_consider)

        f_final_df = self.traindata.windowed_dataset_numpy_hourly(df_to_be_consider, status_cols)
        # print(f_final_df)

        return f_final_df, status_cols

    def process_global_daily_data(self, global_df, out_status, max_date):
        st = time.time()
        global_df = self.traindata.pre_process_global_data_daily(global_df, out_status)
        
        status_cols = list(global_df.columns)
        status_cols.remove('Date')
        add_dates_df = self.traindata.add_missing_dates_without_hours(global_df, max_date)
        # print(add_dates_df)
        global_df = pd.concat([global_df, add_dates_df])
        global_df = global_df.sort_values(['Date']).reset_index(drop=True)
        back_df = self.traindata.add_back_dates_global_daily(global_df, status_cols)
        # print(back_df)
        final_df = pd.concat([global_df, back_df])
        final_df = final_df.sort_values(['Date']).reset_index(drop=True)
        final_df = self.traindata.windowed_dataset_numpy_global_daily(final_df, status_cols)
        # print(final_df)
        return final_df, status_cols

    def training(self, control_name, req_df, max_date):
        try:
            print(control_name)
            if self.pre.useDB:
                ctrl_df = req_df[req_df['Control'] == control_name]
                ctrl_df = ctrl_df.drop(['Control', 'Attribute'], axis=1)
            else:
                ctrl_df = req_df.copy()
                ctrl_df = ctrl_df.drop(['_id','Control', 'Attribute'], axis=1)
            # result_df = ctrl_df.copy()
            ctrl_df.sort_values("Date", inplace=True)
            ctrl_freq = ctrl_df.shape[0]
            result_df_proc = ctrl_df.copy().reset_index().drop(['index'], axis=1)
            final_df, status_cols = self.process_global_daily_data(result_df_proc, self.pre.ctrl_status_col, max_date)
            # print(status_cols)
            final_df = self.traindata.feature_eng_global_daily(final_df, status_cols)
            x_train, y_train, x_test, y_test = self.traindata.train_test_split_global_daily(final_df, status_cols)
            model, mse, mae = self.traindata.build_global_model(x_train, y_train, x_test, y_test, status_cols)
            #print(mse, mae)
            # exit(0)
            # Save model with version
            results = pd.DataFrame(columns=['Control Name', 'Freq', 'Model', 'MSE', 'MAE'])
            results.loc[0] = [control_name, int(ctrl_freq), model, mse, mae]
            # print(results)
            path, version = self.save.save_model_by_control_name(results, control_name+'_daily')
            print(path)
            msg = "Model file successfully created: {}".format(control_name)
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            msg = "Model not trained for control: {}".format(control_name)
            version = 'NA'
        return msg, control_name, version

    def training_daily(self, control_name, req_df):
        try:
            ctrl_df = req_df[req_df['Control'] == control_name]
            ctrl_df = ctrl_df.drop(['Control', 'Attribute'], axis=1)

            result_df = ctrl_df.copy()
            result_df.sort_values("Date", inplace=True)

            ctrl_freq = result_df.shape[0]

            result_df_proc = result_df.copy().reset_index().drop(['index'], axis=1)

            # Preprocessing to Add Missing Values
            data_df, status_cols = self.process_daily_data(result_df_proc)

            # Splitting our Data into Train and Test
            x_train, y_train, x_test, y_test = self.traindata.train_test_split_before_pca_daily(data_df, status_cols)

            # Applying PCA to reduce the Dimensionality
            x_train_data_after_fe, y_train_data_after_fe, x_test_data_after_fe, y_test_data_after_fe  = self.traindata.feature_engg_train_test_data_daily(x_train, y_train, x_test, y_test, status_cols)      
          
            # Training our Model
            model, mse, mae = self.traindata.build_global_model(x_train_data_after_fe, y_train_data_after_fe, x_test_data_after_fe, y_test_data_after_fe, status_cols)

            print("MSE : {}".format(mse))
            print("MAE : {}".format(mae))

            # Save model with version
            results = pd.DataFrame(columns=['Control Name', 'Freq', 'Model', 'MSE', 'MAE'])
            results.loc[0] = [control_name, int(ctrl_freq), model, mse, mae]
            
            path, version = self.save.save_model_by_control_name(results, control_name+'_daily')

            msg = "Model file successfully created: {}".format(control_name)
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            msg = "Model not trained for control: {}".format(control_name)
            version = 'NA'

        return msg, control_name, version

    
    def training_hourly(self, control_name, req_df):
        try:
            ctrl_df = req_df[req_df['Control'] == control_name]
            ctrl_df = ctrl_df.drop(['Control', 'Attribute'], axis=1)

            result_df = ctrl_df.copy()
            result_df.sort_values("Date", inplace=True)

            ctrl_freq = result_df.shape[0]

            result_df_proc = result_df.copy().reset_index().drop(['index'], axis=1)

            # Preprocessing to Add Missing Values
            data_df, status_cols = self.process_hourly_data(result_df_proc)

            # Splitting our Data into Train and Test
            x_train, y_train, x_test, y_test = self.traindata.train_test_split_before_pca(data_df, status_cols)

            # Applying PCA to reduce the Dimensionality
            x_train_data_after_fe, y_train_data_after_fe, x_test_data_after_fe, y_test_data_after_fe  = self.traindata.feature_engg_train_test_data_hourly(x_train, y_train, x_test, y_test, status_cols)            

            # Training our Model
            model, mse, mae = self.traindata.build_global_model(x_train_data_after_fe, y_train_data_after_fe, x_test_data_after_fe, y_test_data_after_fe, status_cols)

            print("MSE : {}".format(mse))
            print("MAE : {}".format(mae))

            # Save model with version
            results = pd.DataFrame(columns=['Control Name', 'Freq', 'Model', 'MSE', 'MAE'])
            results.loc[0] = [control_name, int(ctrl_freq), model, mse, mae]
            
            path, version = self.save.save_model_by_control_name(results, control_name+'_hourly')

            msg = "Model file successfully created: {}".format(control_name)
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            msg = "Model not trained for control: {}".format(control_name)
            version = 'NA'

        return msg, control_name, version


    def train_by_control(self, control_name):
        '''
        Training the data based on the control name

        Parameters
        ----------

        control_name : str
            Name of the control(part)

        '''

        req_df = self.traindata.read_data(control_name)
        max_date = datetime.strptime(str(req_df['Date'].max()), '%Y-%m-%d')
        start_time = time.time()
        msg, _, version = self.training(control_name, req_df, max_date)
        end_time = time.time()
        elapsed_time = round((end_time - start_time), 2)
        print("Time taken: ", elapsed_time)
        return {"message": msg, "model_version": version}, 200

    def train_all_controls(self):
        req_df = self.traindata.read_data()
        print('[INFO] Models to be trained: ', len(set(req_df['Control'])))

        pool = Pool()
        tasks = [*zip(set(req_df['Control']), repeat(req_df))]
        response_result = pool.starmap(self.training_daily, iterable=tasks)
        print(response_result)
        return {"message": "Model file successfully created for the Controls", "model_version": response_result}, 200

    def get_history_data(self, usecase):
        try:
            print("History data-")
            pkl_path = os.path.join(self.pre.FILES_PATH, 'df_' + usecase + '_global.pkl')
            date_point_df = open(pkl_path, 'rb')
            global_df = pickle.load(date_point_df)
            global_df = self.traindata.pre_process_global_data_daily(global_df, self.pre.global_status_col)
            status_cols = list(global_df.columns)
            status_cols.remove('Date')
            add_dates_df = self.traindata.add_missing_dates_without_hours(global_df)
            global_df = pd.concat([global_df, add_dates_df])
            global_df = global_df.sorbt_values(['Date']).reset_index(drop=True)
            back_df = self.traindata.add_back_dates_global_daily(global_df, status_cols)
            history_df = pd.concat([global_df, back_df])
            history_df = history_df.sort_values(['Date']).reset_index(drop=True)
            # result = {'usecase': '', 'history_data': {}}
            result = dict()
            result['usecase'] = usecase
            result['history_data'] = history_df.to_dict('records')
            return result, 200
        except Exception as e:
            import traceback
            print(traceback.format_exc())


    def train_globally_hourly(self, usecase):
        self.traindata.read_data(None,usecase)

        pkl_path = os.path.join(self.pre.FILES_PATH, 'df_' + usecase+'_global.pkl')

        date_point_df = open(pkl_path, 'rb')
        global_df = pickle.load(date_point_df)
        
        freq = global_df.shape[0]

        final_df, status_cols = self.process_hourly_data(global_df)

        # Splitting our Data into Train and Test
        x_train, y_train, x_test, y_test = self.traindata.train_test_split_before_pca(final_df, status_cols)

        # Applying PCA to reduce the Dimensionality
        x_train_data_after_fe, y_train_data_after_fe, x_test_data_after_fe, y_test_data_after_fe  = self.traindata.feature_engg_train_test_data_hourly(x_train, y_train, x_test, y_test, status_cols)            

        # Training our Model
        model, mse, mae = self.traindata.build_global_model(x_train_data_after_fe, y_train_data_after_fe, x_test_data_after_fe, y_test_data_after_fe, status_cols)

        results = pd.DataFrame(columns=['Usecase', 'Freq', 'Model', 'MSE', 'MAE'])        
        #results.loc[0] = [self.pre.usecase, int(freq), model, mse, mae]
        results.loc[0] = [usecase, int(freq), model, mse, mae]
    
        #path, version = self.save.save_model_by_control_name(results, self.pre.usecase+'_hourly')
        path, version = self.save.save_model_by_control_name(results, usecase+'_hourly')
        print(path, version)
        return {"message": "Model file successfully created", "model_version": version}, 200

    def train_globally_daily(self, usecase):
        self.traindata.read_data(None,usecase)
        
        pkl_path = os.path.join(self.pre.FILES_PATH, 'df_' + usecase + '_global.pkl')
     
        date_point_df = open(pkl_path, 'rb')
        global_df = pickle.load(date_point_df)

        # max_date = datetime.strptime(str(global_df['Date'].max()), '%Y-%m-%d')
        freq = global_df.shape[0]

        final_df, status_cols = self.process_daily_data(global_df)

        # Splitting our Data into Train and Test
        x_train, y_train, x_test, y_test = self.traindata.train_test_split_before_pca_daily(final_df, status_cols)

        # Applying PCA to reduce the Dimensionality
        x_train_data_after_fe, y_train_data_after_fe, x_test_data_after_fe, y_test_data_after_fe  = self.traindata.feature_engg_train_test_data_daily(x_train, y_train, x_test, y_test, status_cols)            

        # Training our Model
        model, mse, mae = self.traindata.build_global_model(x_train_data_after_fe, y_train_data_after_fe, x_test_data_after_fe, y_test_data_after_fe, status_cols)

        results = pd.DataFrame(columns=['Usecase', 'Freq', 'Model', 'MSE', 'MAE'])        
        #results.loc[0] = [self.pre.usecase, int(freq), model, mse, mae]
        results.loc[0] = [usecase, int(freq), model, mse, mae]
        
        path, version = self.save.save_model_by_control_name(results, self.pre.usecase+'_daily')
        print(path, version)
        return {"message": "Model file successfully created", "model_version": version}, 200


    def train_by_control_daily(self, control_name):
        '''
        Training the data based on the control name on hourly basis

        Parameters
        ----------

        control_name : str
            Name of the control(part)

        '''

        req_df = self.traindata.read_data(control_name)
        # print(req_df.head())
        start_time = time.time()
        msg, _, version = self.training_daily(control_name, req_df)
        end_time = time.time()
        elapsed_time = round((end_time - start_time), 2)
        print("Time taken: ", elapsed_time)
        return {"message": "Model file successfully created", "model_version": version}, 200

    def train_by_control_hourly(self, control_name):
        '''
        Training the data based on the control name on hourly basis

        Parameters
        ----------

        control_name : str
            Name of the control(part)

        '''

        req_df = self.traindata.read_data(control_name)
        start_time = time.time()
        msg, _, version = self.training_hourly(control_name, req_df)
        end_time = time.time()
        elapsed_time = round((end_time - start_time), 2)
        print("Time taken: ", elapsed_time)
        return {"message": "Model file successfully created", "model_version": version}, 200

    def train_all_controls_hourly(self):
        req_df = self.traindata.read_data()
        # version_dict = dict()
        print('[INFO] Models to be trained: ', len(set(req_df['Control'])))

        pool = Pool()
        tasks = [*zip(set(req_df['Control']), repeat(req_df))]
        response_result = pool.starmap(self.training_hourly, iterable=tasks)
        print(response_result)

        # for control_name in set(req_df['Control']):
        #     print(control_name)
        #     results, version = self.training_hourly(control_name, req_df)
        #     version_dict[control_name] = version
        return {"message": "Model file successfully created for all the Controls", "model_version": response_result}, 200


if __name__ == '__main__':
    train = Train()
    # print(train.train_by_control_daily('TYPE PEDALIER AVG'))
    print("start training")
    #  print(train.train_by_control_daily('BUTEE CAPOT'))
    # print(train.train_by_control_hourly('CAPTEUR PARE CHOC SAMARG'))
    # print(train.train_all_controls())
    # print(train.get_history_data('PRISE BOITIER BEPR'))
    # print(train.get_history_data('conformity'))
    print(train.train_all_controls_hourly())
    # print(train.train_globally_daily('conformity'))
    # print(train.train_globally_hourly('conformity'))
    # print()
    # ETIQUETTE FUMEE, BUTEE CAPOT, BOUCHON VALVE AVG, ECROU ANTIVOL ARG
