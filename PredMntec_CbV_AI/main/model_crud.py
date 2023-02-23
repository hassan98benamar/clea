import os, itertools, re
# from PredictiveMaintenance_ai.main.resources.get_resource_path import get_equip_model_path
from PredMntec_CbV_AI.data.get_resource_path import get_model_path


class ModelCrud:

    '''
    A class to get and delete model files

    Methods
    -------

    get_modelfiles(self,model)
        Returning all available model files based on the input given

    delete_modelfiles(self,model)
        Taking the Model Name and deleting model files related to that

    '''

    def get_modelfiles(self, modelno=None):
        '''
        Returning all available model files based on the input given

        Parameters
        ----------
        
        modelno : str
            Model Name 

        '''

        if modelno is None:
            result = dict()
            files_list = os.listdir(get_model_path())
            if len(files_list) != 0:
                iterator = itertools.groupby(files_list, lambda file: re.search("(.*)(_\d{1,2}\.\d{1})(.*)", file).group(1))
                group_list = []
                version_list = []
                for element, group in iterator:
                    version_list = []
                    for file in list(group):
                        match = re.search("(.*_)([0-9]+\.[0-9])(.*)", file)
                        version_list.append(match.group(2))
                    group_list.append({"model_name": element.replace("~", "/"), "model_version": version_list})
                result["models"] = group_list
            else:
                result["models"] = "No models files for the given Model Name"
            return result, 200
        else:
            model = modelno.replace("/", "~")
            files_list = []
            path = get_model_path()
            for file_name in os.listdir(path):
                if file_name.startswith(model+'_'):
                    match = re.search("(.*_)([0-9]+\.[0-9])(.*)", file_name)
                    files_list.append(match.group(2))

            group_dict = {}
            result = dict()
            group_dict["model_name"] = modelno
            group_dict["model_version"] = files_list
            result["models"] = [group_dict]
            if len(files_list) != 0:
                return result, 200
            else:
                return {"message": "No Model Files Found for the specific Model Name", "model_name": modelno}, 404
    
    def delete_modelfiles(self, modelno):
        '''
        Taking the Model Name and deleting model files related to that
        Parameters
        ----------
        modelno : str
            Model Name 
        '''
        model = modelno.replace("/", "~")
        path = get_model_path()
        count = 0
        for file_name in os.listdir(path):
            if file_name.startswith(model+"_"):
                os.remove(os.path.join(path, file_name))
                count += 1
        if count == 0:
            return {"message": "No Model Files for the given Model Name", "model_name": modelno}, 404
        else:
            return {"message": str(count) + " model files has been deleted for the given Model Name", "model_name": modelno}, 200


if __name__ == "__main__":
    m = ModelCrud()
    print(m.get_modelfiles('CAPTEUR PARE CHOC SAMARG_daily'))
    print(m.delete_modelfiles("TYPE PEDALIER AVG"))
            