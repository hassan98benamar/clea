import os


def get_secure_path_to_conf(file_name):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    base_path = os.path.join(dir_path, "conf", file_name)
    return base_path

