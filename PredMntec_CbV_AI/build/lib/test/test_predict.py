import unittest
from PredMntec_CbV_AI.main.predict import Predict

class TestPredict(unittest.TestCase):

    def test_1_predictByCtrl(self):
        self.predict = Predict()
        res = self.predict.predictByCtrl("TYPE PEDALIER AVG", 10)
        self.assertEqual(res[1], 200)
        self.assertEqual(type(res[0]['predictions'][0]), dict)

        res = self.predict.predictByCtrl("ctrl_name", "1.0")
        self.assertEqual(res[1], 404)
        self.assertEqual(res[0]["message"], 'Invalid Control Name')

    def test_2_predictByCtrl_hourly(self):
        self.predict = Predict()
        res = self.predict.predictByCtrl_hourly("TYPE PEDALIER AVG", 2)
        self.assertEqual(res[1], 200)
        self.assertEqual(type(res[0]['predictions'][0]), dict)

        res = self.predict.predictByCtrl_hourly("ctrl_name", "1.0")
        self.assertEqual(res[1], 404)
        self.assertEqual(res[0]["message"], 'Invalid Control Name')

    def test_3_predict_global_daily(self):
        self.predict = Predict()
        res = self.predict.predict_global_daily("conformity", 10)
        self.assertEqual(res[1], 200)
        self.assertEqual(type(res[0]['predictions'][0]), dict)

        res = self.predict.predict_global_daily("usecase", "1.0")
        self.assertEqual(res[1], 404)
        self.assertEqual(res[0]["message"], 'Invalid Usecase')

    def test_4_predict_global_hourly(self):
        self.predict = Predict()
        res = self.predict.predict_global_hourly("conformity", 1)
        self.assertEqual(res[1], 200)
        self.assertEqual(type(res[0]['predictions'][0]), dict)

        res = self.predict.predict_global_hourly("usecase", "1.0")
        self.assertEqual(res[1], 404)
        self.assertEqual(res[0]["message"], 'Invalid Usecase')
