import json
from typing import List
from flask_restplus import Namespace, reqparse


################################################################################
#                            Configuration Management                             #
################################################################################


class ApiConfig:
    """
    A class used to configure rest api

    ...

    Attributes
    ----------
    file_name : str
        name of configuration file
    Methods
    -------
    get_request_param(request_name)
        return corresponding api query argument

     get_namespace(request_name)
        return corresponding api namespace

     get_parser(request_name)
        return corresponding api parser
    """

    def __init__(self, file_name='conf/rest_config.json'):
        with open(file_name) as json_data:
            self.config = json.load(json_data)

    def get_request_param(self, request_name):
        """return corresponding api query argument

        Parameters
        ----------
        request_name : str
            name of request

        """
        query_param_dict = self.config[request_name]["queryParam"]
        return query_param_dict

    def get_namespace(self, request_name):
        """return corresponding api namespace

        Parameters
        ----------
        request_name : str
            name of request

        """
        api = Namespace(request_name)
        return api

    def get_parser(self, request_name):
        """return corresponding api parser

        Parameters
        ----------
        request_name : str
            name of request

        """
        parser = reqparse.RequestParser()
        param = self.get_request_param(request_name)
        for k in param:
                parser.add_argument(k["paramName"], type=eval(k["paramType"]), location='args',
                                help=k["paramDescription"], required=k["isMandatory"])
        return parser
