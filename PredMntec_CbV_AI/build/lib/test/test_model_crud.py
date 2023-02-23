import unittest
from PredMntec_CbV_AI.main.model_crud import ModelCrud

class TestModel(unittest.TestCase):
    def test_get_modelfiles(self):
        self.crud=ModelCrud()
        res=self.crud.get_modelfiles()
        self.assertEqual(res[1],200)
        self.assertEqual(type(res[0]),dict)

        res1=self.crud.get_modelfiles("CAPTEUR daily")
        self.assertEqual(res1[1],404)
        self.assertEqual(res1[0]["message"],"No Model Files Found for the specific Model Name")

        res2=self.crud.get_modelfiles("test_model")
        self.assertEqual(res2[1],404)
        self.assertEqual(type(res2[0]),dict)

    def test_delete_modelfiles(self):
        self.crud=ModelCrud()
        res=self.crud.delete_modelfiles("test_model")
        self.assertEqual(res[1], 404)
        self.assertEqual(res[0]["message"],"No Model Files for the given Model Name")
