from functools import cached_property
from flask_restplus import Api
from flask import Blueprint
from PredMntec_CbV_Restapp.main.controller.Prediction import api as prediction
from PredMntec_CbV_Restapp.main.controller.Training import api as training
from PredMntec_CbV_Restapp.main.controller.Images import api as image
from PredMntec_CbV_Restapp.main.controller.Records import api as record

import os

dir_path = os.path.dirname(os.path.realpath(__file__))
base_path = os.path.join(dir_path, 'resources')
__location__ = os.path.join(os.getcwd())
print(dir_path)
print(base_path)
print(__location__)

blueprint = Blueprint('api', __name__)

api = Api(blueprint,
          title='API REST',
          version='1.0',
          description='REST API to Predict Maintenance'
          )

#add request namespace
api.add_namespace(prediction, path="/predict")
api.add_namespace(training, path="/train")
api.add_namespace(image, path='/images')
api.add_namespace(record, path="/records/<regex('.*'):ctrl_name>")