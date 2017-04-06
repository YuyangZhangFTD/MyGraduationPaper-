"""
    Collaborative Filter
"""
import pandas as pd
import numpy as np
from scipy import sparse
import cf_fun as cf
import RecTool as rt


file_name = "ratings.csv"
rate_m, test_data = rt.file_read(file_name)



