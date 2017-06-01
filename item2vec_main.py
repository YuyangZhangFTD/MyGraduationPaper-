"""
    item2vec with skip gram
"""
import pandas as pd
import numpy as np
from scipy import sparse
import RecTool as rt
import item2vec_skipgram_fun as fun

file_path = "input/ml-latest-small/"
# file_path = 'input/ml-20m/'
file_name = file_path + "ratings.csv"
rate_m, test_data, user, item  = rt.file_read(file_name)
onehot_dict = fun.count_dict(rate_m, item)


rec = {}
print("I'm running!")
for es in [50,75,100,125,150,200,250,300,400,500]:
    rec[es] = {}
    for ws in [1,2,3]:
        embedding_mat, loss_xy, onehot_dict = fun.train(rate_m, user, item,\
                    embedding_size=es, iter_n=50000, window_size=ws,\
                    batch_size = 100, count_th=5, plot_loss=False, method='skip_gram')
        print("Finish train!")
        print(embedding_mat.shape)
        test_hat, skip_n = fun.pred(rate_m, test_data, embedding_mat, onehot_dict, user)
# print("I have finished!")
        rmse = rt.loss_rmse(test_hat, test_data, skip=skip_n)
        rmae = rt.loss_rmae(test_hat, test_data, skip=skip_n)

        print(rmse)
        print(rmae)
print(rec)


