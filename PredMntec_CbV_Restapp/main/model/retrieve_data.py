from typing import Dict, Any
import logging
import traceback
import json
from flask import send_file
from PredMntec_CbV_AI.main.predict import Predict
from PredMntec_CbV_AI.main.train import Train
from PredMntec_CbV_AI.main.model_crud import ModelCrud
from PredMntec_CbV_Restapp.main.service.log_user_action import user_action_logging
from PredMntec_CbV_Restapp.main.util.read_setting import get_secure_path_to_conf
################################################################################
#                            PredictiveMaintenance AI Model                                  #
################################################################################

template = "An exception of type {0} occurred."


def exception_hander(ex):
    message = template.format(type(ex).__name__, ex.args)
    logging.error(traceback.format_exc())
    logging.error(message)

try:
    path = get_secure_path_to_conf('api_mapping.json')
    with open(path) as json_data:
        config = json.load(json_data)

except Exception as ex:
    message = template.format(type(ex).__name__, ex.args)
    logging.error(traceback.format_exc())
    logging.error(message)

def get_mapped_args(request_type, args):

    try:
        map_dict = config[request_type]
        map_args: Dict[str, Any] = {}
        for m in map_dict:
            if m["type"] == "list":
                if m["map"] in args:
                    map_args[m["map"]] = args[m["name"]].split(',')
                else:
                    map_args[m["map"]] = {}
            else:
                if m["map"] == "version" and "version" not in args.keys():
                    continue
                map_args[m["map"]] = args[m["name"]]
        return map_args

    except ValueError as vex:
        logging.error(vex)
        return None

def get_model_files(model=None):
    """Get data through PredictiveMaintenance AI

    Parameters
    ----------
    args : list([str])
        requests query param

    """
    logging.info({'modelname':model})
    model_class=ModelCrud()
    try:
        if model == None:
            response,code = model_class.get_modelfiles()
        else:
            response,code = model_class.get_modelfiles(model)
        user_action_logging("model", model)
            

    except AttributeError as aex:
        exception_message = template.format(type(aex).__name__, aex.args)
        logging.error(exception_message)
        return {"status": "failed", "message": "An exception of type AttributeError occurred."}, 500

    except ValueError as vex:
        exception_message = template.format(type(vex).__name__, vex.args)
        logging.error(vex)
        logging.error(exception_message)
        return {"status": "failed", "message": str(vex)}, 500

    return {"status": "success", "message": response}, code

def delete_model_files(modelname):
    """Delete model files from PredictiveMaintenance AI

    Parameters
    ----------
    args : list([str])
        requests query param

    """
    model_class=ModelCrud()
    logging.info({'modelname':modelname})
    try:
        response,code = model_class.delete_modelfiles(modelname)
        user_action_logging("model", modelname)
            

    except AttributeError as aex:
        exception_message = template.format(type(aex).__name__, aex.args)
        logging.error(exception_message)
        return {"status": "failed", "message": "An exception of type AttributeError occurred."}, 500

    except ValueError as vex:
        exception_message = template.format(type(vex).__name__, vex.args)
        logging.error(vex)
        logging.error(exception_message)
        return {"status": "failed", "message": str(vex)}, 500

    return {"status": "success", "message": response}, code

def get_prediction(args,ctrl_name,period):
    """Get data through PredictiveMaintenance CbV

    Parameters
    ----------
    args : list([str])
        requests query param

    """
    pre=Predict()
    map_args = get_mapped_args('predict', args)
    map_args["period"]=period
    logging.info(map_args)
    try:
        if map_args == None:
            return {"status": "failed", "message": "Internal Error"}, 500
        else:
            if "version" in map_args.keys():
                response,code = pre.predictByCtrl(ctrl_name, int(period), map_args["version"])
            else:
                response,code = pre.predictByCtrl(ctrl_name, int(period))
            user_action_logging("predict", *[v for v in map_args.values()])      

    except AttributeError as aex:
        exception_message = template.format(type(aex).__name__, aex.args)
        logging.error(exception_message)
        return {"status": "failed", "message": "An exception of type AttributeError occurred."}, 500

    except KeyError as kex:
        exception_message = template.format(type(kex).__name__, kex.args)
        logging.error(exception_message)
        return {"status": "failed", "message": "An exception of type KeyError occurred."}, 500

    except ValueError as vex:
        exception_message = template.format(type(vex).__name__, vex.args)
        logging.error(vex)
        logging.error(exception_message)
        return {"status": "failed", "message": str(vex)}, 500

    return {"status": "success", "message": response}, code

def get_hourly_prediction(args,ctrl_name,period):
    """Get data through PredictiveMaintenance CbV

    Parameters
    ----------
    args : list([str])
        requests query param

    """
    pre=Predict()
    map_args = get_mapped_args('predict', args)
    map_args["period"]=period
    logging.info(map_args)
    try:
        if map_args == None:
            return {"status": "failed", "message": "Internal Error"}, 500
        else:
            if "version" in map_args.keys():
                response,code = pre.predictByCtrl_hourly(ctrl_name, int(period), map_args["version"])
            else:
                response,code = pre.predictByCtrl_hourly(ctrl_name, int(period))
            user_action_logging("predict", *[v for v in map_args.values()])      

    except AttributeError as aex:
        exception_message = template.format(type(aex).__name__, aex.args)
        logging.error(exception_message)
        return {"status": "failed", "message": "An exception of type AttributeError occurred."}, 500

    except KeyError as kex:
        exception_message = template.format(type(kex).__name__, kex.args)
        logging.error(exception_message)
        return {"status": "failed", "message": "An exception of type KeyError occurred."}, 500

    except ValueError as vex:
        exception_message = template.format(type(vex).__name__, vex.args)
        logging.error(vex)
        logging.error(exception_message)
        return {"status": "failed", "message": str(vex)}, 500

    return {"status": "success", "message": response}, code

def get_pred_global_hourly(self,usecase,period):
    pre=Predict()
    map_args={}
    map_args["period"]=period
    logging.info(map_args)
    try:
        if map_args == None:
            return {"status": "failed", "message": "Internal Error"}, 500
        else:
            response,code = pre.predict_global_hourly(usecase, int(period))
            user_action_logging("predict", *[v for v in map_args.values()])         

    except AttributeError as aex:
        exception_message = template.format(type(aex).__name__, aex.args)
        logging.error(exception_message)
        return {"status": "failed", "message": "An exception of type AttributeError occurred."}, 500

    except KeyError as kex:
        exception_message = template.format(type(kex).__name__, kex.args)
        logging.error(exception_message)
        return {"status": "failed", "message": "An exception of type KeyError occurred."}, 500

    except ValueError as vex:
        exception_message = template.format(type(vex).__name__, vex.args)
        logging.error(vex)
        logging.error(exception_message)
        return {"status": "failed", "message": str(vex)}, 500

    return {"status": "success", "message": response}, code

def get_pred_global_daily(self,usecase,period):
    pre=Predict()
    map_args={}
    map_args["period"]=period
    logging.info(map_args)
    try:
        if map_args == None:
            return {"status": "failed", "message": "Internal Error"}, 500
        else:
            response,code = pre.predict_global_daily(usecase, int(period))
            user_action_logging("predict", *[v for v in map_args.values()])         

    except AttributeError as aex:
        exception_message = template.format(type(aex).__name__, aex.args)
        logging.error(exception_message)
        return {"status": "failed", "message": "An exception of type AttributeError occurred."}, 500

    except KeyError as kex:
        exception_message = template.format(type(kex).__name__, kex.args)
        logging.error(exception_message)
        return {"status": "failed", "message": "An exception of type KeyError occurred."}, 500

    except ValueError as vex:
        exception_message = template.format(type(vex).__name__, vex.args)
        logging.error(vex)
        logging.error(exception_message)
        return {"status": "failed", "message": str(vex)}, 500

    return {"status": "success", "message": response}, code

def get_all_hourly_predictions(self,period):
    """Get data through PredictiveMaintenance CvB

    Parameters
    ----------
    args : list([str])
        requests query param

    """
    pre=Predict()
    map_args={}
    map_args["period"]=period
    logging.info(map_args)
    try:
        if map_args == None:
            return {"status": "failed", "message": "Internal Error"}, 500
        else:
            response,code = pre.predict_all_hourly(int(period))
            user_action_logging("predict", *[v for v in map_args.values()])         

    except AttributeError as aex:
        exception_message = template.format(type(aex).__name__, aex.args)
        logging.error(exception_message)
        return {"status": "failed", "message": "An exception of type AttributeError occurred."}, 500

    except KeyError as kex:
        exception_message = template.format(type(kex).__name__, kex.args)
        logging.error(exception_message)
        return {"status": "failed", "message": "An exception of type KeyError occurred."}, 500

    except ValueError as vex:
        exception_message = template.format(type(vex).__name__, vex.args)
        logging.error(vex)
        logging.error(exception_message)
        return {"status": "failed", "message": str(vex)}, 500

    return {"status": "success", "message": response}, code

def get_all_predictions(self,period):
    """Get data through PredictiveMaintenance CvB

    Parameters
    ----------
    args : list([str])
        requests query param

    """
    pre=Predict()
    map_args={}
    map_args["period"]=period
    logging.info(map_args)
    try:
        if map_args == None:
            return {"status": "failed", "message": "Internal Error"}, 500
        else:
            response,code = pre.predict_all(int(period))
            user_action_logging("predict", *[v for v in map_args.values()])         

    except AttributeError as aex:
        exception_message = template.format(type(aex).__name__, aex.args)
        logging.error(exception_message)
        return {"status": "failed", "message": "An exception of type AttributeError occurred."}, 500

    except KeyError as kex:
        exception_message = template.format(type(kex).__name__, kex.args)
        logging.error(exception_message)
        return {"status": "failed", "message": "An exception of type KeyError occurred."}, 500

    except ValueError as vex:
        exception_message = template.format(type(vex).__name__, vex.args)
        logging.error(vex)
        logging.error(exception_message)
        return {"status": "failed", "message": str(vex)}, 500

    return {"status": "success", "message": response}, code


def do_training(self,ctrl_name):

    """Train the model from PredictiveMaintenance CbV

        Parameters
        ----------
        args : list([str])
            requests query param

    """
    train_class=Train()
    logging.info({'ctrl_name':ctrl_name})
    try:
        response,code = train_class.train_by_control(ctrl_name)
        user_action_logging("model", ctrl_name)
            

    except AttributeError as aex:
        exception_message = template.format(type(aex).__name__, aex.args)
        logging.error(exception_message)
        return {"status": "failed", "message": "An exception of type AttributeError occurred."}, 500

    except ValueError as vex:
        exception_message = template.format(type(vex).__name__, vex.args)
        logging.error(vex)
        logging.error(exception_message)
        return {"status": "failed", "message": str(vex)}, 500

    return {"status": "success", "message": response}, code

def do_training_all_controls_hourly(self):
    train_class=Train()
    try:
        response,code = train_class.train_all_controls_hourly()
        user_action_logging("model","all Controls")
    except AttributeError as aex:
        exception_message = template.format(type(aex).__name__, aex.args)
        logging.error(exception_message)
        return {"status": "failed", "message": "An exception of type AttributeError occurred."}, 500

    except ValueError as vex:
        exception_message = template.format(type(vex).__name__, vex.args)
        logging.error(vex)
        logging.error(exception_message)
        return {"status": "failed", "message": str(vex)}, 500

    return {"status": "success", "message": response}, code

def do_training_all(self):
    """Train the model from PredictiveMaintenance CbV

    Parameters
    ----------
    args : list([str])
        requests query param

    """
    train_class=Train()
    try:
        response,code = train_class.train_all_controls()
        user_action_logging("model","all Controls")
            

    except AttributeError as aex:
        exception_message = template.format(type(aex).__name__, aex.args)
        logging.error(exception_message)
        return {"status": "failed", "message": "An exception of type AttributeError occurred."}, 500

    except ValueError as vex:
        exception_message = template.format(type(vex).__name__, vex.args)
        logging.error(vex)
        logging.error(exception_message)
        return {"status": "failed", "message": str(vex)}, 500

    return {"status": "success", "message": response}, code

def do_training_globally_hourly(self, usecase):
    train_class=Train()
    logging.info({'usecase':usecase})
    try:
        response,code = train_class.train_globally_hourly(usecase)
        user_action_logging("model", usecase)
    except AttributeError as aex:
        exception_message = template.format(type(aex).__name__, aex.args)
        logging.error(exception_message)
        return {"status": "failed", "message": "An exception of type AttributeError occurred."}, 500

    except ValueError as vex:
        exception_message = template.format(type(vex).__name__, vex.args)
        logging.error(vex)
        logging.error(exception_message)
        return {"status": "failed", "message": str(vex)}, 500

    return {"status": "success", "message": response}, code

def do_training_globally_daily(self, usecase):
    train_class=Train()
    logging.info({'usecase':usecase})
    try:
        response,code = train_class.train_globally_daily(usecase)
        user_action_logging("model", usecase)
    except AttributeError as aex:
        exception_message = template.format(type(aex).__name__, aex.args)
        logging.error(exception_message)
        return {"status": "failed", "message": "An exception of type AttributeError occurred."}, 500

    except ValueError as vex:
        exception_message = template.format(type(vex).__name__, vex.args)
        logging.error(vex)
        logging.error(exception_message)
        return {"status": "failed", "message": str(vex)}, 500

    return {"status": "success", "message": response}, code

def do_training_by_control_hourly(self, ctrl_name): 
    train_class=Train()
    logging.info({'ctrl_name':ctrl_name})
    try:
        response,code = train_class.train_by_control_hourly(ctrl_name)
        user_action_logging("model", ctrl_name)
    except AttributeError as aex:
        exception_message = template.format(type(aex).__name__, aex.args)
        logging.error(exception_message)
        return {"status": "failed", "message": "An exception of type AttributeError occurred."}, 500

    except ValueError as vex:
        exception_message = template.format(type(vex).__name__, vex.args)
        logging.error(vex)
        logging.error(exception_message)
        return {"status": "failed", "message": str(vex)}, 500

    return {"status": "success", "message": response}, code
    
def get_all_predictions_by_count(self,period):
    """Get data through PredictiveMaintenance CvB

    Parameters
    ----------
    args : list([str])
        requests query param

    """
    pre=Predict()
    map_args={}
    map_args["period"]=period
    logging.info(map_args)
    try:
        if map_args == None:
            return {"status": "failed", "message": "Internal Error"}, 500
        else:
            response,code = pre.get_date_json(int(period))
            user_action_logging("predict", *[v for v in map_args.values()])         

    except AttributeError as aex:
        exception_message = template.format(type(aex).__name__, aex.args)
        logging.error(exception_message)
        return {"status": "failed", "message": "An exception of type AttributeError occurred."}, 500

    except KeyError as kex:
        exception_message = template.format(type(kex).__name__, kex.args)
        logging.error(exception_message)
        return {"status": "failed", "message": "An exception of type KeyError occurred."}, 500

    except ValueError as vex:
        exception_message = template.format(type(vex).__name__, vex.args)
        logging.error(vex)
        logging.error(exception_message)
        return {"status": "failed", "message": str(vex)}, 500

    return {"status": "success", "message": response}, code

def get_records(ctrl_name):
    """Train the model from PredictiveMaintenance AI

    Parameters
    ----------
    args : list([str])
        requests query param

    """
    train_class=Train()
    try:
        response,code = train_class.get_history_data(ctrl_name)
        user_action_logging("record",ctrl_name)
            

    except AttributeError as aex:
        exception_message = template.format(type(aex).__name__, aex.args)
        logging.error(exception_message)
        return {"status": "failed", "message": "An exception of type AttributeError occurred."}, 500

    except ValueError as vex:
        exception_message = template.format(type(vex).__name__, vex.args)
        logging.error(vex)
        logging.error(exception_message)
        return {"status": "failed", "message": str(vex)}, 500

    return {"status": "success", "message": response}, code

if __name__ == '__main__':
    print(get_records('conformity'))