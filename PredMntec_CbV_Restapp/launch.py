import logging
import json
import sys
from PredMntec_CbV_Restapp.main import create_app
from PredMntec_CbV_Restapp.main.util.read_setting import get_secure_path_to_conf
from waitress import serve


def run():
    """Runs flask app"""
    config = {}
    path = get_secure_path_to_conf("server_config.json")
    try:
        if len(sys.argv) == 3:
            config["host"] = sys.argv[1]
            config["port"] = sys.argv[2]
        else:
            with open(path) as json_data:
                config = json.load(json_data)
        logging.info("running on {}:{}".format(config["host"], config["port"]))
        app = create_app()
        serve(app,host=config["host"],port=config["port"])
    except KeyError as kex:
        exception_message = "An exception of type {0} occurred.".format(type(kex).__name__, kex.args)
        logging.error(exception_message)


if __name__ == "__main__":
    run()
