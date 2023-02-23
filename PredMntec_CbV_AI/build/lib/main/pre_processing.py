#!/usr/bin/env python3

import pandas as pd
import numpy as np
import json
import os
from tqdm import tqdm
from pymongo import MongoClient
from PredMntec_CbV_AI.data.get_resource_path import get_path

class PreProcess:
    '''A Class for reading and formatting the data

    Methods
    =======
    load_files(self)
        Loading all the jsons files for the desired usecase

    get_data(self,data,detect_list)
        Extract useful data from json

    read_data(self):
        Read the json files and create the dataframe for desired usecase
    
    '''
    # FILES_PATH = r'C:\Users\harssaxe\Desktop\PredMntec CbV\data'
    FILES_PATH = get_path()
    folder = 'mu_conformity'
    usecase = 'conformity'
    range_hours = range(0, 24)
    ctrl_status_col = 'Status'
    global_status_col = 'Status'
    useDB = False

    def load_files(self):
        # Loading all the jsons files for the desired usecase
        json_files = [pos_json for pos_json in os.listdir(os.path.join(self.FILES_PATH,self.folder))
                    if pos_json.endswith('.json') and 
                    (pos_json.startswith('V') or pos_json.startswith('W'))]
                    #   (not pos_json.startswith('ERROR') and not pos_json.startswith('MISSING'))]
        return json_files

    def get_data(self, data, detect_list):
        # Extract useful data from jsons 
        order_timestamp = data['creationDate']
        order_date = order_timestamp.split('T')[0]
        for detection in data['result']['detections']:
            ai_status = detection['aiStatus']
            attribute = detection['reference']['attribute']
            control = detection['reference']['control']
            
            if 'userStatus' in detection:
                user_status = detection['userStatus']
            else:
                user_status = ''

            tp = [attribute, control, ai_status, user_status, order_date, order_timestamp]
            detect_list.append(tp)
        return detect_list

    def get_data_global(self, data, detect_list):
        # get global status details of the usecase
        order_timestamp = data['creationDate']
        order_date = order_timestamp.split('T')[0]
        global_status = np.where(data['result']['globalExpertStatus'] != '', data['result']['globalExpertStatus'],
                                 np.where(data['result']['globalUserStatus'] != '', data['result']['globalUserStatus'], data['result']['globalAiStatus']))
        tp = [order_date, order_timestamp, global_status]
        detect_list.append(tp)
        return detect_list

    def read_data(self, status_type):
        # Read the json files and create the dataframe for desired usecase
        json_files = self.load_files()
        detect_list = []
        gs = []
        us = []
        for f in tqdm(json_files, desc="Reading files"):
            with open(os.path.join(self.FILES_PATH, self.folder, f)) as json_file:
                try:
                    data = json.load(json_file)
                    gs.append(data['globalState'])
                    us.append(data['useCase'])
                    if 'globalState' not in data:
                        continue
                    if data['useCase'] != self.usecase:
                        continue
                    if data['globalState'] != 'PRED':
                        continue
                    if status_type == 'global':
                        detect_list = self.get_data_global(data, detect_list)
                        df = pd.DataFrame(detect_list, columns=['Date', 'Timestamp', 'GlobalStatus'])
                        path = os.path.join(self.FILES_PATH, 'df_'+self.pre.usecase+'_global.pkl')
                        df.to_pickle(path)
                    else:
                        detect_list = self.get_data(data, detect_list)
                        df = pd.DataFrame(detect_list, columns=['Attribute', 'Control', 'Ai Status', 'User Status', 'Date', 'Timestamp'])
                except Exception as e:
                    pass
        #     print(df)
        return df

    def save_as_pickle(self, df, file_name):
        path = os.path.join(self.FILES_PATH, file_name)
        df.to_pickle(path)
        print("Pickle file created", path)



def main():
    pre = PreProcess()
    df = pre.read_data('local')
    req_df = df.copy()
    req_df.drop(['Ai Status', 'User Status'], axis=1)
    req_df['Status'] = np.where(req_df['User Status'] != '', req_df['User Status'], req_df['Ai Status'])
    req_df = req_df.drop(['Ai Status', 'User Status'], axis=1)
    req_df['Status'] = req_df['Status'].replace(['NOT FOUND'], 'NOT_FOUND')
    req_df['Status Code'], label = pd.factorize(req_df['Status'])
    # req_df_copy = req_df.copy()
    # req_df = req_df.drop(['Status'], axis=1)

    global_df = pre.read_data('global')
    global_df['GlobalStatus'] = global_df['GlobalStatus'].map(lambda x: x.item())

    if pre.useDB:
        # Save in MongoDB
        client = MongoClient('localhost', 27017)
        db = client['cbv']
        control_collection = db['Controls']
        global_collection = db['Global']

        records_to_insert = req_df.to_dict("records")
        control_collection.insert_many(records_to_insert)

        records_to_insert_g = global_df.to_dict("records")
        global_collection.insert_many(records_to_insert_g)
    else:
        # Save as pickle file
        pre.save_as_pickle(req_df, 'df_' + pre.usecase+'_global.pkl')
        pre.save_as_pickle(req_df, 'df_'+pre.usecase+'.pkl')


if __name__ == '__main__':
    main()
