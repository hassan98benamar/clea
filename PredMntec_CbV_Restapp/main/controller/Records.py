from flask import request
from flask_restplus import Resource
from PredMntec_CbV_Restapp.main.model.retrieve_data import get_records
from PredMntec_CbV_Restapp.main.service.api_parameter import ApiConfig
from PredMntec_CbV_Restapp.main.util.read_setting import get_secure_path_to_conf

request_name = "records"

path = get_secure_path_to_conf("rest_config.json")
apiConfig = ApiConfig(path)
api = apiConfig.get_namespace(request_name)

result = api.schema_model('Records', {
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
              "history_data": {
                "type": "array",
                "items": {
                  "type": "object",
                  "properties": {
                    "date": {
                      "type": "string",
                      "format": "date"
                    },
                    "status": {
                        "type": "string"
                      }
                  }
                }
              }
            }
          }
        }
    }
)

@api.route("/")
class Record(Resource):
    """
    A class used to handle Prediction By EquipId and Date

    Methods
    -------
    get()
        Handle get request
    """
    @api.response(200,'success',result)
    def get(self,ctrl_name):
        """take GET request information and return the predicted values based on that (?)"""
        return get_records(ctrl_name)

