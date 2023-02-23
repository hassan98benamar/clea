import os
from flask import Flask
from flask_cors import CORS
from PredMntec_CbV_Restapp.main.config import config_by_name
from PredMntec_CbV_Restapp.main.service import blueprint
from werkzeug.routing import BaseConverter


class RegexConverter(BaseConverter):
        def __init__(self, url_map, *items):
                super(RegexConverter, self).__init__(url_map)
                self.regex = items[0]
            
def create_app():
    """Create flask app object

    Parameters
    ----------
    config_name : string
        configuration type

    """
    if "PredictiveMaintenance" in os.environ:
        config_name = str(os.environ["PredictiveMaintenance"])
    else:
        config_name = "prod"
    app = Flask(__name__)
    CORS(app)
    app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
    app.url_map.converters['regex'] = RegexConverter
    app.config.from_object(config_by_name[config_name])
    app.register_blueprint(blueprint)

    return app
