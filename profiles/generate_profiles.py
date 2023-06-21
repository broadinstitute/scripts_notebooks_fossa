"""
Functions to process profiles from single-cells or well-aggregated information

@fefossa
"""

import pycytominer
import seaborn as sns
import matplotlib.pyplot as plt
import operator
import pandas as pd
import numpy as np
import requests
import random

def stringToBool(correlation_input):
    if correlation_input == 'yes':
       return True
    elif correlation_input == 'no':
       return False
    else:
      raise ValueError