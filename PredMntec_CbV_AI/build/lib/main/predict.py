import os
from datetime import datetime
import pandas as pd
import numpy as np
import time
from PredMntec_CbV_AI.main.pre_processing import PreProcess
from PredMntec_CbV_AI.main.processing_train_data import ProcessTrain
from PredMntec_CbV_AI.main.save_model import Save
from PredMntec_CbV_AI.main.train import Train
from PredMntec_CbV_AI.data.get_resource_path import get_model_path
import pickle
from multiprocessing import Pool
from itertools import repeat

import json

def get_model(ctrl_name):
    try:
        model_name = ""
        file_list = []
        # path = r'C:\Users\harssaxe\Desktop\PredMntec CbV\PredMntec_CbV_Ai\saved_model'
        path = get_model_path()
        for file in os.listdir(path):
            if file.startswith(ctrl_name.replace("/", "~")+"_"):
                file_list.append(os.path.join(path, file))
        if len(file_list) == 0:
            model_name = None
        else:
            model_name = max(file_list, key=os.path.getctime)
        return model_name
    except Exception as e:
        print(e)

class Predict:

    '''
    A class to make predictions

    Methods
    -------
    '''
    pre = PreProcess()
    traindata = ProcessTrain()
    save = Save()
    train = Train()
    
    
    def predict_vals_hourly(self, pred_df, window_size, model_data, dim_red, max_date, status_cols):
        st = time.time()
        model = model_data['Model']

        window_size = window_size

        vals = pred_df.tail(1).values[0]
        last_date = vals[-2]
        
        days_diff = (max_date-last_date).days
        window_size += days_diff
        slice_len = (len(status_cols))+2

        start_date = last_date + pd.DateOffset(days=1)        
        end_date = start_date + pd.DateOffset(days=window_size)
        
        range_date = pd.date_range(start_date, end_date).tolist()
        range_date = [x for x in range_date if x.weekday() != 6]
        last_row = [x for x in vals[:-slice_len]]
        
        if dim_red:
            dim_red_df = self.traindata.feature_eng_hourly(pred_df, status_cols)
            last_row_red = dim_red_df.tail(1).values[0][:-slice_len]

        for i in range(len(range_date)):
            for h in self.pre.range_hours:
                index = pred_df.index[-1]
                if dim_red:                    
                    pred = model.predict(np.array([last_row_red]))
                else:
                    pred = model.predict([last_row])[0]
                pred = np.round(pred, decimals=0)                
                date = range_date[i]
    
                for k in range(len(pred[0])):
                    last_row = last_row + [int(pred[0][k])]
                last_row = last_row + [date.strftime('%Y-%m-%d')] + [str(h)]
                
                pred_df.loc[index+1] = last_row

                last_row = last_row[1:-slice_len+1]
                if dim_red:
                    dim_red_df = self.traindata.feature_eng_hourly(pred_df, status_cols)
                    last_row_red = dim_red_df.tail(1).values[0][:-slice_len]

        return pred_df, len(range_date)


    def predict_vals_daily(self, pred_df, window_size, model_data, dim_red, max_date, status_cols):
        st = time.time()
        model = model_data['Model']
        
        vals = pred_df.tail(1).values[0]

        last_date = vals[-1]
        
        days_diff = (max_date-last_date).days
        window_size += days_diff

        slice_len = (len(status_cols))+1
        start_date = last_date + pd.DateOffset(days=1)
        end_date = start_date + pd.DateOffset(days=window_size)

        range_date = pd.date_range(start_date, end_date).tolist()
        range_date = [x for x in range_date if x.weekday() != 6]

        last_row = [x for x in vals[:-slice_len]]
        if dim_red:
           
            dim_red_df = self.traindata.feature_eng_daily(pred_df, status_cols)
            last_row_red = dim_red_df.tail(1).values[0][:-slice_len]

        for i in range(len(range_date)):
            index = pred_df.index[-1]
            if dim_red:
                pred = model.predict(np.array([last_row_red]))
            else:
                pred = model.predict([last_row])[0]
            pred = np.round(pred, decimals=0)
           
            date = range_date[i]
            
            for k in range(len(pred[0])):
                last_row = last_row + [int(pred[0][k])]

            last_row = last_row + [date.strftime('%Y-%m-%d')]            
            pred_df.loc[index+1] = last_row

            # Req last_row
            last_row = last_row[1:-slice_len+1]
            if dim_red:
                dim_red_df = self.traindata.feature_eng_daily(pred_df, status_cols)
                last_row_red = dim_red_df.tail(1).values[0][:-slice_len]
        
        return pred_df[days_diff:], len(range_date)



    def predict_vals_global_daily(self, pred_df, window_size, model_data, dim_red, max_date, status_cols):
        model = model_data['Model']
        # print(status_cols)
        # print(pred_df)
        vals = pred_df.tail(1).values[0]
        last_date = vals[-1]
        #last_date = datetime.strptime(str(last_date), '%Y-%m-%d')
        days_diff = (max_date-last_date).days
        window_size += days_diff
        slice_len = (len(status_cols))+1
        start_date = last_date + pd.DateOffset(days=1)
        end_date = start_date + pd.DateOffset(days=window_size)
        range_date = pd.date_range(start_date, end_date).tolist()
        range_date = [x for x in range_date if x.weekday() != 6]
        last_row = [x for x in vals[:-slice_len]]
        if dim_red:
            #         pred_df.loc[pred_df.shape[0]] = last_row + [0, range_date[0]]
            # # print("-------------------------------------")
            # print(pred_df)
            dim_red_df = self.traindata.feature_eng_daily(pred_df, status_cols)
            last_row_red = dim_red_df.tail(1).values[0][:-slice_len]

        for i in range(len(range_date)):
            index = pred_df.index[-1]
            if dim_red:
                #             print("inside dim", np.array([last_row_red]))
                pred = model.predict(np.array([last_row_red]))
            else:
                pred = model.predict([last_row])[0]
            pred = np.round(pred, decimals=0)
            # print(pred)
            date = range_date[i]
            # print("bfr", len(last_row))
            for k in range(len(pred[0])):
                last_row = last_row + [int(pred[0][k])]
            last_row = last_row + [date.strftime('%Y-%m-%d')]
            # print("aftr", len(last_row))
            pred_df.loc[index+1] = last_row

            # Req last_row
            last_row = last_row[1:-slice_len+1]
            if dim_red:
                dim_red_df = self.traindata.feature_eng_global_daily(pred_df, status_cols)
                last_row_red = dim_red_df.tail(1).values[0][:-slice_len]
        # print(len(range_date), days_diff, window_size)
        return pred_df[days_diff:], len(range_date)

    def process_control_predict_daily(self, ctrl_name, req_df, dim_red, window_size, status_col, model_version=None):
        try:
            print(ctrl_name)
            st = time.time()
            max_date = datetime.strptime(str(req_df['Date'].max()), '%Y-%m-%d')

            model_name = get_model(ctrl_name+'_daily')
            # print("model_name", model_name)
            model_data = self.save.get_model_metrics(model_name, ctrl_name+'_daily')
            # print(model_data)

            print("MSE", model_data['MSE'])
            print("MAE", model_data['MAE'])

            req_df = req_df.drop_duplicates(subset=['Date', 'Control', 'Status'])

            ctrl_df = req_df[req_df['Control'] == ctrl_name]
            ctrl_df = ctrl_df.drop(['Control', 'Attribute'], axis=1)
            result_df = ctrl_df.copy()
            result_df.sort_values("Date", inplace=True)

            data_df, status_cols = self.train.process_daily_data_for_prediction(result_df)
            # print(data_df)

            pred_df, window_range = self.predict_vals_daily(data_df, window_size, model_data, dim_red, max_date, status_cols)
            # print("ff", window_range)
            
            t_cols = ['t' + str(x) for x in range(-len(status_cols)+1, 1)]
            out_cols = t_cols + ['Date']
            out_df = pred_df[out_cols][-window_range:]
            # print(out_df)
            for c in t_cols:
                out_df[c] = np.where(out_df[c] < 0, 0, out_df[c])
            # print(t_cols, status_cols)
            out_df.rename(columns=dict(zip(t_cols, status_cols)), inplace=True)
            result = dict()
            if status_col == 'Status':
                result['ctrl_name'] = ctrl_name
            else:
                result['usecase'] = self.pre.usecase
            result['predictions'] = out_df.to_dict('records')
            return result
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            result = {'message': "Model data cannot make prediction for the given Control",
                      'ctrl_name': ctrl_name}
        return result

    def predictByCtrl(self, ctrl_name, window_size=30, model_version=None):
        '''
        Making Predictions based on Control and formatting the output

        Parameters
        ----------

        ctrl_name : str
            Control name for which prediction has to be made

        model_version : str
            File version in which trained data is saved

        '''

        start_time = time.time()
        dim_red = True

        req_df = self.traindata.read_data(ctrl_name)
        if ctrl_name in req_df['Control'].values:
            try:
                result = self.process_control_predict_daily(ctrl_name, req_df, dim_red, window_size-1, None)
                end_time = time.time()
                print("Time taken: ", (end_time - start_time))
                result['elapsed'] = end_time - start_time
                return result, 200
            except Exception as e:
                import traceback
                print(traceback.format_exc())
                result = {'message': "Model data cannot make prediction for the given Control",
                          'ctrl_name': ctrl_name}
                return result, 400
        else:
            result = {'message': "Invalid Control Name",
                    'ctrl_name': ctrl_name}
            return result, 404

    def predict_all(self, window_size=1):
        # window_size = 5
        dim_red = True
        req_df = self.traindata.read_data()
        start_time = time.time()
        response_result = []
        print('[INFO] Predictions to be done: ', len(set(req_df['Control'])))
        # Multiprocessing
        print("multiprocessing")
        pool = Pool()
        tasks = [*zip(set(req_df['Control']), repeat(req_df), repeat(dim_red), repeat(window_size), repeat(self.pre.ctrl_status_col))]
        response_result = pool.starmap(self.process_control_predict_daily, iterable=tasks)
        # status_list = ['PASS', 'FAIL', 'NOT_FOUND', 'UNEXPECTED', 'DEFICIT']
        # for r in response_result:
        #     if 'message' in r:
        #         #         response_result.remove(r)
        #         continue
        #     for st in status_list:
        #         if st not in r['predictions'][0]:
        #             r['predictions'][0][st] = 0.0
        # for r in response_result:
        #     if 'message' in r:
        #         response_result.remove(r)

        end_time = time.time()
        # print(response_result)

        # for c in response_result['message']:
        #     if 'message' in c:
        #         response_result['message'].remove(c)
        print("Time taken: ", (end_time - start_time))
        # response_result['elapsed'] = end_time - start_time
        return response_result, 200

    def get_date_json(self, period):
        response_result, _ = self.predict_all(period)
        count_dict = dict()

        for each in response_result:
            status_dict = {'No maintenance': 0,
                           'PASS': 0,
                           'UNEXPECTED': 0,
                           'NOT_FOUND': 0,
                           'FAIL': 0,
                           'DEFICIT': 0}
            for pred in each['predictions']:
                if pred['Date'] not in count_dict:
                    try:
                        count_dict[pred['Date']] = {}
                        count_dict[pred['Date']][pred['Status']] = 1
                    except:
                        pass
                elif pred['Date'] in count_dict and pred['Status'] not in count_dict[pred['Date']]:
                    count_dict[pred['Date']][pred['Status']] = 1
                else:
                    try:
                        count_dict[pred['Date']][pred['Status']] += 1
                    except:
                        pass

        result = []
        for date in count_dict.keys():
            temp = dict()
            temp["date"] = date
            temp["predictions"] = count_dict[date]
            result.append(temp)

        return result, 200

    def predict_global_hourly(self, usecase, window_size=1):
        start_time = time.time()
        dim_red = True

        try:
            pkl_path = os.path.join(self.pre.FILES_PATH, 'df_' + usecase+'_global.pkl')
            date_point_df = open(pkl_path, 'rb')
            global_df = pickle.load(date_point_df)
        except Exception as e:
            result = {'message': "Invalid Usecase",
                      'usecase': usecase}
            return result, 404
        if not global_df.empty:
            try:                
                # Loading a Trained Model
                model_name = get_model(usecase + '_hourly')
                model_data = self.save.get_model_metrics(model_name, usecase + '_hourly')
                
                print("MSE", model_data['MSE'])
                print("MAE", model_data['MAE'])

                max_date = datetime.strptime(str(global_df['Date'].max()), '%Y-%m-%d')

                # Preprocessing
                data_df, status_cols = self.train.process_hourly_data_for_prediction(global_df)

                # Prediction
                pred_df, window_range = self.predict_vals_hourly(data_df, window_size, model_data, dim_red, max_date, status_cols)
                
                pred_df['Hour'] = pred_df['Hour'].astype(int)
                hour_diff = max(pred_df['Hour']) - min(pred_df['Hour']) + 1
                
                t_cols = ['t' + str(x) for x in range(-len(status_cols)+1, 1)]
                out_cols = t_cols + ['Date', 'Hour']
                out_df = pred_df[out_cols][-(window_size)*hour_diff:]
                for c in t_cols:
                    out_df[c] = np.where(out_df[c] < 0, 0, out_df[c])
                out_df.rename(columns=dict(zip(t_cols, status_cols)), inplace=True)
        
                end_time = time.time()

                print("Time taken: ", round((end_time - start_time)/60, 2))
                result = dict()
                result['usecase'] = self.pre.usecase
                result['predictions'] = out_df.to_dict('records')
                end_time = time.time()
                result['elapsed'] = round((end_time - start_time)/60, 2)
                return result, 200

            except Exception as e:
                import traceback
                print(traceback.format_exc())
                result = {'message': "Model data cannot make prediction for the given Case",
                        'usecase': usecase}
                return result, 400
        else:
            result = {'message': "Invalid Usecase",
                      'usecase': usecase}
        return result, 404

    def predict_global_daily(self, usecase, window_size=1):
        start_time = time.time()
        dim_red = True

        try:
            ppkl_path = os.path.join(self.pre.FILES_PATH, 'df_' + usecase+'_global.pkl')
            date_point_df = open(pkl_path, 'rb')
            global_df = pickle.load(date_point_df)
        except Exception as e:
            result = {'message': "Invalid Usecase ",
                      'usecase': usecase}
            return result, 404
        if not global_df.empty:
            try:
                max_date = datetime.strptime(str(global_df['Date'].max()), '%Y-%m-%d')
                model_name = get_model(usecase + '_daily')
                model_data = self.save.get_model_metrics(model_name, usecase + '_daily')

                print("MSE", model_data['MSE'])
                print("MAE", model_data['MAE'])

                data_df, status_cols = self.train.process_daily_data_for_prediction(global_df)
                pred_df, window_range = self.predict_vals_daily(data_df, window_size, model_data, dim_red, max_date, status_cols)
                
                t_cols = ['t' + str(x) for x in range(-len(status_cols)+1, 1)]
                out_cols = t_cols + ['Date']
                out_df = pred_df[out_cols][-window_range:]
                
                for c in t_cols:
                    out_df[c] = np.where(out_df[c] < 0, 0, out_df[c])
                
                out_df.rename(columns=dict(zip(t_cols, status_cols)), inplace=True)
                
                end_time = time.time()
                print("Time taken: ", round((end_time - start_time)/60, 2))
                result = dict()
                result['usecase'] = self.pre.usecase
                result['predictions'] = out_df.to_dict('records')
                end_time = time.time()
                result['elapsed'] = round((end_time - start_time)/60, 2)
                return result, 200
            except Exception as e:
                # import traceback
                # print(traceback.format_exc())
                result = {'message': "Model data cannot make prediction for the given Case",
                        'usecase': usecase}
                return result, 400
        else:
            result = {'message': "Invalid Usecase",
                      'usecase': usecase}
        return result, 404


    def process_control_predict_hourly(self, ctrl_name, req_df, dim_red, window_size, model_version=None):
        try:
            print(ctrl_name)
            st = time.time()
            max_date = datetime.strptime(str(req_df['Date'].max()), '%Y-%m-%d')
    
            model_name = get_model(ctrl_name+'_hourly')
            
            model_data = self.save.get_model_metrics(model_name, ctrl_name+'_hourly')
            
            print("MSE", model_data['MSE'])
            print("MAE", model_data['MAE'])

            req_df = req_df.drop_duplicates(subset=['Date', 'Control', 'Status'])
            
            ctrl_df = req_df[req_df['Control'] == ctrl_name]
            ctrl_df = ctrl_df.drop(['Control', 'Attribute'], axis = 1)

            result_df = ctrl_df.copy()
            result_df.sort_values("Date", inplace=True)
            result_df_proc = result_df.copy().reset_index().drop(['index'], axis=1)
            
            data_df, status_cols = self.train.process_hourly_data_for_prediction(result_df_proc)

            pred_df, window_range = self.predict_vals_hourly(data_df, window_size, model_data, dim_red, max_date, status_cols)

            pred_df['Hour'] = pred_df['Hour'].astype(int)

            hour_diff = max(pred_df['Hour']) - min(pred_df['Hour']) + 1
            t_cols = ['t' + str(x) for x in range(-len(status_cols)+1, 1)]
            out_cols = t_cols + ['Date', 'Hour']
        
            out_df = pred_df[out_cols][-(window_size)*hour_diff:]
            for c in t_cols:
                out_df[c] = np.where(out_df[c] < 0, 0, out_df[c])
            out_df.rename(columns=dict(zip(t_cols, status_cols)), inplace=True)

            result = dict()
            result['ctrl_name'] = ctrl_name
            result['predictions'] = out_df.to_dict('records')
            return result
        
        except Exception as e:
            import traceback
            result = {'message': "Model data cannot make prediction for the given Control : Error from process_control_predict_hourly function",
                    'ctrl_name': ctrl_name}
            return result

    def predictByCtrl_hourly(self, ctrl_name, window_size=1, model_version=None):
        print(ctrl_name)
        start_time = time.time()
        dim_red = True

        req_df = self.traindata.read_data(ctrl_name)
        if ctrl_name in req_df['Control'].values:
            try:
                result = self.process_control_predict_hourly(ctrl_name, req_df, dim_red, window_size, None)
                end_time = time.time()
                print("Time taken: ", (end_time - start_time))
                result['elapsed'] = end_time - start_time
                return result, 200
            except Exception as e:
                import traceback
                print(traceback.format_exc())
                result = {'message': "Model data cannot make prediction for the given Control Name : Error From predictByCtrl_hourly Function",
                        'ctrl_name': ctrl_name}
                return result, 400
        else:
            result = {'message': "Invalid Control Name",
                      'ctrl_name': ctrl_name}
            return result, 404

    def predictByCtrl_daily(self, ctrl_name, window_size=1, model_version=None):
        '''
        Making Predictions based on Control and formatting the output

        Parameters
        ----------

        ctrl_name : str
            Control name for which prediction has to be made

        model_version : str
            File version in which trained data is saved

        '''
        print(ctrl_name)
        start_time = time.time()
        dim_red = True

        req_df = self.traindata.read_data(ctrl_name)
        if ctrl_name in req_df['Control'].values:
            try:
                result = self.process_control_predict_daily(ctrl_name, req_df, dim_red, window_size, self.pre.ctrl_status_col, None)
                end_time = time.time()
                print("Time taken: ", (end_time - start_time))
                result['elapsed'] = end_time - start_time
                return result, 200
            except Exception as e:
                import traceback
                print(traceback.format_exc())
                result = {'message': "Model data cannot make prediction for the given Control",
                          'ctrl_name': ctrl_name}
                return result, 400
        else:
            result = {'message': "Invalid Control Name",
                    'ctrl_name': ctrl_name}
            return result, 404


    def predict_all_hourly(self, window_size=1):
        # window_size = 60
        dim_red = True

        req_df = self.traindata.read_data()
        response_result = []
        print('[INFO] Predictions to be done: ', len(set(req_df['Control'])))
        # Multiprocessing
        print("multiprocessing")
        pool = Pool()
        tasks = [*zip(set(req_df['Control']), repeat(req_df), repeat(dim_red), repeat(window_size), repeat(None))]
        response_result = pool.starmap(self.process_control_predict_hourly, iterable=tasks)

        # Without Multiprocessing
        # for ctrl_name in set(req_df.head(50)['Control']):
        #     print(ctrl_name)
        #     try:
        #         start_time = time.time()
        #         result = self.process_control_predict_hourly(ctrl_name, req_df, dim_red, window_size, None)
        #         end_time = time.time()
        #         print("Time taken: ", (end_time - start_time))
        #         result['elapsed'] = round((end_time - start_time)/60, 2)
        #         response_result.append(result)
        # 
        #     except Exception as e:
        #         import traceback
        #         result = {'message': "Model data cannot make prediction for the given Control",
        #                 'ctrl_name': ctrl_name}
        #         response_result.append(result)
        #         return result, 400
        return response_result, 200


if __name__ == '__main__':
    pred = Predict()
    s_t = time.time()

    # print(pred.predictByCtrl_daily('CAPTEUR PARE CHOC SAMARG', 60))
    # print(pred.get_date_json(5))
    # print(pred.predictByCtrl_daily('TYPE PEDALIER AVG', 60))
    # print(pred.predictByCtrl_daily('BUTEE CAPOT', 3))
    # print(pred.predictByCtrl_hourly('TYPE PEDALIER AVG', 3))
    # print(pred.predictByCtrl_hourly('CAPTEUR PARE CHOC SAMARG', 3))
    # print(get_model('CAPTEUR PARE CHOC SAMARG'))
    print(pred.predict_all_hourly(3))
    # BUTEE CAPOT DROITE, ASPECT GRILLE PARE CHOC AVANT
    # print(pred.predict_all(3))
    # print(pred.predict_global_daily('conformity', 3))
    # print(pred.predict_global_hourly('conformity', 3))
    print(time.time() - s_t)
