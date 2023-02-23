from flask import request
from flask_restplus import Resource
from PredMntec_CbV_Restapp.main.model.retrieve_data import get_prediction, get_all_predictions, get_all_predictions_by_count, get_pred_global_hourly, get_all_hourly_predictions, get_hourly_prediction, get_pred_global_daily
from PredMntec_CbV_Restapp.main.service.api_parameter import ApiConfig
from PredMntec_CbV_Restapp.main.util.read_setting import get_secure_path_to_conf

request_name = "predict"

path = get_secure_path_to_conf("rest_config.json")
apiConfig = ApiConfig(path)
api = apiConfig.get_namespace(request_name)
parser = apiConfig.get_parser(request_name)

result = api.schema_model('Predictions', {
    "type": "object",
        "properties": {
          "status": {
            "type": "string"
          },
          "message": {
            "type": "object",
            "properties": {
              "ctrl_name": {
                "type": "string"
              },
              "predictions": {
                "type": "array",
                "items": {
                  "type": "object",
                  "properties": {
                    "accuracy": {
                      "type": "number",
                      "format": "float"
                    },
                    "date": {
                      "type": "string",
                      "format": "date"
                    },
                    "maintenance": {
                      "type": "boolean"
                    }
                  }
                }
              },
              "elapsed": {
                "type": "number",
                "format": "float"
              }
            }
          },
          "graph": {
            "type": "image"
          }
        }
    }
)

result_all = api.schema_model('PredictionsAll', {
    "type": "object",
        "properties": {
          "status": {
            "type": "string"
          },
          "message": {
            "type": "array",
            "items": {
              "properties": {
                "ctrl_name": {
                  "type": "string"
                },
                "predictions": {
                  "type": "array",
                  "items": {
                    "type": "object",
                    "properties": {
                      "accuracy": {
                        "type": "number",
                        "format": "float"
                      },
                      "date": {
                        "type": "string",
                        "format": "date"
                      },
                      "work_order": {
                        "type": "string"
                      }
                    }
                  }
                }
              }
            }
          }
        }
}
)

@api.route("/<regex('.*'):ctrl_name>/<period>")
class Prediction(Resource):
    """
    A class used to handle Prediction By Control Name

    Methods
    -------
    get()
        Handle get request
    """
    @api.expect(parser)
    @api.response(200,'success',result)
    def get(self,ctrl_name,period):
        """take GET request information and return the predicted values based on the crtl name and the period"""
        return get_prediction(request.args,ctrl_name,period)

@api.route("_hourly/<regex('.*'):ctrl_name>/<period>")
class HourlyPrediction(Resource):
    """
    A class used to handle Prediction By Control Name

    Methods
    -------
    get()
        Handle get request
    """
    @api.expect(parser)
    @api.response(200,'success',result)
    def get(self,ctrl_name,period):
        """take GET request information and return the predicted values hourly based on the crtl name and the period"""
        return get_hourly_prediction(request.args,ctrl_name,period)

@api.route('all/<period>')
class AllPrediciton(Resource):
    """
    A class used to handle Prediction By Period size for all controls

    Methods
    -------
    get()
        Handle get request
    """
    # @api.expect(parser)
    @api.response(200,'success',result_all)
    def get(self,period):
        """take GET request information and return all the predicted values based on ctrl name model trained"""
        return get_all_predictions(request.args,period)

@api.route('count/<period>')
class AllPredicitonByCount(Resource):
    """
    A class used to handle Prediction By Period size for all controls

    Methods
    -------
    get()
        Handle get request
    """

    # @api.expect(parser)
    @api.response(200,'success',result_all)
    def get(self,period):
        """take GET request information and return the predicted values based on that (?)"""
        return get_all_predictions_by_count(request.args,period)

@api.route("_global_hourly/<regex('.*'):usecase>/<period>")
class PredictionGlobalHourly(Resource): 
    """
    A class used to handle Prediction By Control Name

    Methods
    -------
    get()
        Handle get request
    """
    @api.response(200,'success',result)
    def get(self,usecase,period):
        """take GET request information and return the predicted values global hourly based by the usecase and the period"""
        return get_pred_global_hourly(request.args,usecase,period)

@api.route("_global_daily/<regex('.*'):usecase>/<period>")
class PredictionGlobalDaily(Resource): 
    """
    A class used to handle Prediction By Control Name

    Methods
    -------
    get()
        Handle get request
    """
    @api.response(200,'success',result)
    def get(self,usecase,period):
        """take GET request information and return the predicted values global daily based by the usecase and the period"""
        return get_pred_global_daily(request.args,usecase,period)
  
@api.route('_all_hourly/<period>')
class AllHourlyPrediciton(Resource):
    """
    A class used to handle Prediction By Period size for all controls

    Methods
    -------
    get()
        Handle get request
    """
    # @api.expect(parser)
    @api.response(200,'success',result_all)
    def get(self,period):
        """take GET request information and return the predicted values hourly by the period"""
        return get_all_hourly_predictions(request.args,period)