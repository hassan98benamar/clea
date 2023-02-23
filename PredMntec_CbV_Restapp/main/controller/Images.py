from re import A
from flask import request, send_from_directory
from flask_restplus import Resource
from PredMntec_CbV_Restapp.main.service.api_parameter import ApiConfig
from PredMntec_CbV_Restapp.main.util.read_setting import get_secure_path_to_conf

request_name = "images"

path = get_secure_path_to_conf("rest_config.json")
apiConfig = ApiConfig(path)
api = apiConfig.get_namespace(request_name)

@api.route("/<file_name>")
class Images(Resource):
    """
    A class used to handle Prediction By EquipId and Date

    Methods
    -------
    get()
        Handle get request
    """

    def get(self,file_name):
        """take GET request information and return the predicted values based on that (?)"""
        return send_from_directory(r'C:\Users\aessakip\AppData\Roaming\Python\Python38\site-packages\PredictiveMaintenance_ai\main\pipeline\Graphs', file_name, cache_timeout=0)
