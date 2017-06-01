"""
    Collaborative Filter
"""
import pandas as pd
import numpy as np
from scipy import sparse
import cf_fun as cf
import RecTool as rt

file_path = "input/ml-latest-small/"
file_name = file_path + "ratings.csv"
rate_m, test_data, user, item  = rt.file_read(file_name)

print("I'm running!")
test_hat = cf.pred(rate_m, test_data, user, item, para_sim=cf.sim_distance, user_based=True)
print("I have finished!")
for i in range(len(test_data)):
    if i % 500 == 0:
        hat = test_hat[i]
        true = test_data[i]
        print("user:  "+str(hat[0])+"   movie:  "+str(hat[1]))
        print("true value -->  "+str(true[2])+" estimated value -->   "+str(hat[2]))

print(rt.loss_rmse(test_hat, test_data))
