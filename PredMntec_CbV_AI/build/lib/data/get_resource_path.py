#!/usr/bin/env python3

import os
def get_path():
    here = os.path.abspath(os.path.dirname(__file__))
    return here

def get_model_path():
    here = os.path.abspath(os.path.dirname(__file__))
    base_path = os.path.join(here, 'saved_model')
    return base_path