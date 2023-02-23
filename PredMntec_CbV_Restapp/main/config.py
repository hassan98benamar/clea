import os
from logging.config import fileConfig
from PredMntec_CbV_Restapp.main.util.read_setting import get_secure_path_to_conf

path = get_secure_path_to_conf("logging.cfg")

class Config:
    SECRET_KEY = os.getenv('SECRET_KEY', 'my_precious_secret_key')
    DEBUG = False


class DevelopmentConfig(Config):
    print(path)
    fileConfig(path)
    DEBUG = True


class TestingConfig(Config):
    DEBUG = True
    TESTING = True


class ProductionConfig(Config):
    fileConfig(path)
    DEBUG = False


config_by_name = dict(
    dev=DevelopmentConfig,
    test=TestingConfig,
    prod=ProductionConfig
)

key = Config.SECRET_KEY

