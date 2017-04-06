"""
    Evaluation Functions in Collaborative Filter 
"""
import random as rd
import numpy as np
import pandas as pd
from scipy import sparse


def file_read(para_name, para_splitprecent=0.9):
    """
        Read rating matrix, and split the file into trian set and test set. 
        Return rating matrix and test data.
    :param para_name:           file name
    :param para_splitprecent:   the percent of train set and test set, default 0.9
    :return:                    rating matrix and test data
    """
    test_data = []
    file_path = "input/ml-latest-small/"
    file_name = file_path+para_name
    rate = pd.read_csv(file_name)
    del rate["timestamp"]
    user_num = max(rate['userId'])
    item_num = max(rate['movieId'])
    rate = rate.values
    rate_m = sparse.dok_matrix((user_num, item_num))
    for vec in rate:
        if rd.random() > para_splitprecent:
            test_data.append([int(vec[0]-1), int(vec[1]-1), vec[2]])    # test data
        else:
            rate_m[int(vec[0]-1), int(vec[1]-1)] = vec[2]     # array start from 0  
    return rate_m, test_data
    

def file_read_2(para_name, para_splitprecent=0.9, para_max_user=100000, para_max_item=200000):
    """
        When file is to large to load in memory, use this function.
        Read rating matrix, and split the file into trian set and test set. 
        Return rating matrix and test data.
    :param para_name:           file name
    :param para_splitprecent:   the percent of train set and test set, default 0.9
    :return:                    rating matrix and test data
    """
    test_data = []
    file_path = "input/ml-latest-small/"
    file_name = file_path+para_name
    rate_m = sparse.dok_matrix((para_max_user, para_max_item))
    with open(file_name) as f:
        f.readline()
        while True:
            tmp = f.readline().split(',')
            if len(tmp) < 2:
                break
            vec = [int(tmp[0]), int(tmp[1]), float(tmp[2])]         
            if rd.random() > para_splitprecent:
                test_data.append([int(vec[0]-1), int(vec[1]-1), vec[2]])    # test data
            else:
                rate_m[int(vec[0]-1), int(vec[1]-1)] = vec[2]     # array start from 0
    return rate_m, test_data


def get_dict_user(para_m):
    """
        Return a dict based on user.
        Record rating information of one user.
    :param para_m:      rating matrix
    :return:            a dict     
    """
    num_user, num_item = para_m.shape
    user = {}
    for i in range(num_user):
        user[i] = {}
        for j in range(num_item):
            if a[i,j] > 0:
                user[i][j] = a[i,j]
    return user


def get_dict_item(para_m):
    """
        Return a dict based on item.
        Record rating information of one item.
    :param para_m:      rating matrix
    :return:            a dict     
    """
    num_user, num_item = para_m.shape
    item = {}
    for j in range(num_item):
        item[j] = {}
        for i in range(num_user):
            if a[i,j] > 0:
                item[j][i] = a[i,j]
    return item


def loss_rmse(para_hat, para_true):
    """
        The RMSE loss.
        The format of input vector:
            user_id, item_id, rate
    :param para_hat:    estimated value
    :param para_true:   true value
    :return:            the rmse loss
    """
    loss = 0
    n = len(para_hat)
    for ii in range(n):
        loss += pow(para_hat[ii][3] - para_true[ii][3], 2)
    return loss/n