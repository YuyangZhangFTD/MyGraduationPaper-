"""
    Matrix Factorization
"""
import pandas as pd
import numpy as np
from scipy import sparse
import mf_fun as mf
import RecTool as rt

file_path = "input/ml-latest-small/"
file_name = file_path + "ratings.csv"
rate_m, test_data, user, item  = rt.file_read(file_name)
print("I'm running!")
test_hat = mf.pred(rate_m, test_data, 8, epoch_n=10, learning_rate=0.005, with_bias=False)
print("I have finished!")
print(rt.loss_rmse(test_hat, test_data))
print(rt.loss_rmae(test_hat, test_data))
