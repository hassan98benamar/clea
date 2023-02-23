import time
from datetime import datetime
import re
import pandas as pd
import os, pickle
from zipfile import ZipFile
from os.path import basename
from PredMntec_CbV_AI.data.get_resource_path import get_model_path

class Save:
    '''
    A class to save and load the model developed

    Methods
    -------

    save_model(results,control_name)
        Saving the model in a zip file based on the trained data

   '''

    def save_model_by_control_name(self, results, control_name):

        '''
        Saving the model in a zip file based on the trained data
         
        Parameters
        ----------

        results : DataFrame
            Data Frame that is trained data

        control_name: str
            Control_name/ Usecase name

        '''

        # logs_dir = r'C:\Users\harssaxe\Desktop\PredMntec CbV\PredMntec_CbV_Ai\saved_model'
        logs_dir = get_model_path()
        print("[INFO] - Saving Model")
        file_list=[]
        for file_name in os.listdir(logs_dir):
            if file_name.startswith(control_name.replace("/", "~")+'_'):
                file_list.append(os.path.join(logs_dir, file_name))
        if len(file_list) == 0:
            version = 1.0
        else:
            # print(file_list)
            latest_file = max(file_list, key=os.path.getctime)
            # print(latest_file)
            match = re.search(r'(.*_)([0-9]+\.[0-9])(.*)', latest_file)
            version = round(float(match.group(2))+0.1, 1)
        
        path_m = os.path.join(logs_dir, '{}_{}.zip'.format(control_name.replace("/", "~"), version))
        if 'Control Name' in results:
            for ctrl in results['Control Name']:
                # val = ctrl
                path = os.path.join(logs_dir, ctrl.replace("/", "~")+".pkl")
                results.loc[results['Control Name'] == ctrl].to_pickle(path)
                # df.to_csv(os.path.join(logs_dir, ctrl+'_window_data.csv'))
        elif 'Usecase' in results:
            for uc in results['Usecase']:
                # val = uc
                path = os.path.join(logs_dir, uc.replace("/", "~")+".pkl")
                results.loc[results['Usecase'] == uc].to_pickle(path)
                # df.to_csv(os.path.join(logs_dir, uc+'_window_data.csv'))
        pkl_files = [os.path.join(logs_dir, x) for x in os.listdir(logs_dir) if x.endswith('pkl')]

        # pkl_files.append(os.path.join(logs_dir, val +'_window_data.csv'))
        # print(pkl_files)
        with ZipFile(path_m, 'w') as zipObj:
            for filename in pkl_files:
                # First parameter is complete path
                # Second parameter is the base file name
                zipObj.write(filename, basename(filename))
        
        # Removing all tmp_pkl files
        list(map(os.remove, pkl_files))
        return os.path.basename(path_m), version


    def get_model_metrics(self, model_name, ctrl_name=None):

        '''
        Load the model based on the model_name and ctrl_name

        Parameters
        ----------

        model_name : str
            Model file name 

        ctrl_name : str
            Control Name/ Usecase

        '''
        
        try:
            # logs_dir = r'C:\Users\harssaxe\Desktop\PredMntec CbV\PredMntec_CbV_Ai\saved_model'
            logs_dir = get_model_path()
            model_path = os.path.join(logs_dir, model_name)
            zf = ZipFile(model_path, 'r')
            # If condition will never be true for model inference
            if ctrl_name == None:
                mod_df = pd.DataFrame(columns=['Control Name', 'Freq', 'Model', 'Recall', 'Accuracy', 'RMSE'])
                for name in zf.namelist():
                    df = pickle.load(zf.open(name, 'r'))
                    mod_df = mod_df.append(df, ignore_index=True)
                zf.close()
                print("The log of the model is -->", mod_df.tail(1)['Model'].values[0])
                print("Captured Equip", mod_df.shape)
                if 'Accuracy' in mod_df.columns:
                    print("Mean of Accuracy", mod_df[mod_df['Accuracy'] != -1]['Accuracy'].mean())
                if 'Recall' in mod_df.columns:
                    print("Mean of Recall", mod_df[mod_df['Recall'] != -1]['Recall'].mean())
                print("Train Err Count -", mod_df[mod_df['Model'] == 'TRAIN_ERR'].shape)
                print("Train Err ratio -", mod_df[mod_df['Model'] == 'TRAIN_ERR'].shape[0] / mod_df.shape[0])
                return None
            else:
                ctrl_name_upd = ctrl_name.replace("/", "~")
                # print("zip list", zf.namelist())
                # print("heeeeere", ctrl_name_upd.split('_')[0]+'.pkl')
                # if ctrl_name_upd+'.pkl' in zf.namelist():
                # print(zf.namelist())
                if ctrl_name_upd.split('_')[0]+'.pkl' in zf.namelist():
                    mod_df = pickle.load(zf.open(ctrl_name_upd.split('_')[0]+'.pkl', 'r'))
                    if 'Control Name' in mod_df.columns:
                        model_data = mod_df[mod_df['Control Name'] == ctrl_name.split('_')[0]].to_dict('records')[0]
                    elif 'Usecase' in mod_df.columns:
                        model_data = mod_df[mod_df['Usecase'] == ctrl_name.split('_')[0]].to_dict('records')[0]

                    # print(ctrl_name_upd.split('_')[0]+'_window_data.csv')
                    # if ctrl_name_upd.split('_')[0]+'_window_data.csv' in zf.namelist():
                    #     data_df = pd.read_csv(zf.open(ctrl_name_upd.split('_')[0]+'_window_data.csv'))
                        # print(data_df)
                    # # Extracting Build Paramsg
                    # mod_df = pickle.load(zf.open('build_params.pkl','r'))
                    # zf.close()
                    # build_params = mod_df[mod_df['Equip'] == 'build_params'].to_dict('records')[0]['Model']
                else:
                    return 'Model data does not exist'
            return model_data
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            print(e)

if __name__ == '__main__':
    save = Save()
    s_t = time.time()
    print(save.get_model_metrics('CAPTEUR PARE CHOC SAMARG_hourly_1.6.zip', 'CAPTEUR PARE CHOC SAMARG_hourly'))
    print(time.time() - s_t)
