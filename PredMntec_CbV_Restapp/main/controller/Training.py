from flask import request
from flask_restplus import Resource
from PredMntec_CbV_Restapp.main.model.retrieve_data import get_model_files, delete_model_files, do_training, do_training_all, do_training_by_control_hourly, do_training_all_controls_hourly, do_training_globally_daily, do_training_globally_hourly
from PredMntec_CbV_Restapp.main.service.api_parameter import ApiConfig
from PredMntec_CbV_Restapp.main.util.read_setting import get_secure_path_to_conf

request_name = "train"

path = get_secure_path_to_conf("rest_config.json")
apiConfig = ApiConfig(path)
api = apiConfig.get_namespace(request_name)
# parser = apiConfig.get_parser(request_name)

get_result=api.schema_model('ModelList',{
    "type": "object",
        "properties": {
          "status": {
            "type": "string"
          },
          "message": {
            "type": "object",
            "properties": {
              "models": {
                "type": "array",
                "items": {
                  "type": "object",
                  "properties": {
                    "modelname": {
                      "type": "string"
                    },
                    "model_version": {
                      "type": "array",
                      "items": {
                        "type": "string",
                        "format": "\\d+\\.\\d"
                      }
                    }
                  }
                }
              }
            }
          }
        }
})

post_result=api.schema_model('PostResponse',{
        "type": "object",
        "properties": {
          "status": {
            "type": "string"
          },
          "message": {
            "type": "object",
            "properties": {
              "message": {
                "type": "string"
              },
              "model_version": {
                "type": "string"
              }
            }
          }
        }
    })

post_all_result=api.schema_model('PostAllResponse',{
        "type": "object",
        "properties": {
          "status": {
            "type": "string"
          },
          "message": {
            "type": "object",
            "properties": {
              "message": {
                "type": "string"
              },
              "model_version": {
                "type": "array",
                  "items": {
                    "type": "string",
                    "format": "\\d+\\.\\d"
                  }
              }
            }
          }
        }
    })

delete_result=api.schema_model('DeleteResponse',{
    "type": "object",
        "properties": {
          "status": {
            "type": "string"
          },
          "message": {
            "type": "object",
            "properties": {
              "message": {
                "type": "string"
              },
              "modelname": {
                "type": "string"
              }
            }
          }
        }
})

@api.route("_delete_model/<regex('.*'):modelname>")
class DeleteModel(Resource):
    """
    A class used to perform crud operation
    Methods
    -------
    get()
        Handle get request
    ------
    delete()
        Handle delete request
    """
    @api.response(200,"success",delete_result)
    def delete(self,modelname):
        """take DELETE request information and delete model files by the ctrl name"""
        return delete_model_files(modelname)

@api.route("_get_specific_model/<regex('.*'):modelname>")
class GetSpecificModel(Resource):
    """
    A class used to perform crud operation
    Methods
    get()
        Handle get request
    """
    @api.response(200,"success",get_result)
    def get(self,modelname):
        """take GET request information and return model files by the ctrl name"""
        return get_model_files(modelname)

@api.route("_get_all_model")
class GetAllModel(Resource):
    """
    A class used to perform crud operation
    Methods
    get()
        Handle get request
    """
    @api.response(200,"success",get_result)
    def get(self):
        """return list of all the available models along with its version"""
        return get_model_files()

@api.route("/<regex('.*'):ctrl_name>")
class Training(Resource):
  # @api.expect(parser)
  @api.response(200,'success',post_result)
  def get(self,ctrl_name):
      """take GET request information and return the train model by the crtl name"""
      return do_training(request.args,ctrl_name)

@api.route("/")
class Training_all(Resource):
    """
    A class used to handle Training for all

    Methods
    -------
    get()
        Handle get request
    """
    # @api.expect(parser)
    @api.response(200,'success',post_all_result)
    def get(self):
        """take GET request information and return all model trained"""
        return do_training_all(request.args)

@api.route("_ctrl_hourly/<regex('.*'):ctrl_name>")
class Training_hourly(Resource):
  @api.response(200,'success',post_result)
  def get(self,ctrl_name):
      """take GET request information and return the model trained hourly by the crtl name"""
      return do_training_by_control_hourly(request.args,ctrl_name)

@api.route("_globally_daily/<regex('.*'):usecase>")
class Training_global_daily(Resource):
  @api.response(200,'success',post_result)
  def get(self,usecase):
      """take GET request information and return return all model trained globally daily by the usecase"""
      return do_training_globally_daily(request.args,usecase)

@api.route("_ctrl_all_hourly")
class Training_all_hourly(Resource):
    @api.response(200,'success',post_all_result)
    def get(self):
        """take GET request information and return the all model trained hourly"""
        return do_training_all_controls_hourly(request.args)

@api.route("_globally_hourly/<regex('.*'):usecase>")
class Training_global_hourly(Resource):
  @api.response(200,'success',post_result)
  def get(self,usecase):
      """take GET request information and return all model trained globally hourly by the usecase"""
      return do_training_globally_hourly(request.args,usecase)