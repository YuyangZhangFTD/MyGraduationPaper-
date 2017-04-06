"""
    Functions in Baseline
"""
import numpy as np
import RecTool as rt 


def get_mu(para_m):
    """
        Get the average of all data
    """
    return sum(para_m.values())/para_m.getnnz()

def get_ave_bu(para_m, para_mu):
    """
        Get the bias of each user by average
    :param para_m:      rating matrix
    :param para_mu:     average of rating matrix
    :return:            a list     
    """
    num_user, __ = para_m.shape
    return [sum(para_m[i,:].values())/para_m[i,:].getnnz()-para_mu for i in range(num_user)]


def get_ave_bi(para_m, para_mu):
    """
        Get the bias of each item by average
    :param para_m:      rating matrix
    :param para_mu:     average of rating matrix
    :return:            a list     
    """
    __, num_item = para_m.shape
    bi_list = []
    for i in range(num_item):
        total = para_m[:,i].getnnz()
        if total == 0:
            bi_list.append(0)
        else:
            bi_list.append(sum(para_m[:,i].values())/total-para_mu)
    return bi_list


def get_bu_bi(para_m, para_mu, epoch_n=100, learning_rate=0.01, reg=0.1):
    """
        Use SGD to get bu and bi
        Minimize the loss:
            \min_{b*} \sum{(r_{ij}-\mu-b_u-b_i)^2 + \lambda_1(\sum{b_u}^2 + \sum{b_i}^2)}
    """
    num_user, num_item = para_m.shape
    bu = np.zeros([num_user, 1]) 
    bi = np.zeros([num_item, 1])
    for epoch_i in range(epoch_n):
        for k,v in para_m.items():
            error = (v - para_mu - bu[k[0]] - bi[k[1]])
            bu[k[0]] += learning_rate * (error - reg * bu[k[0]])
            bi[k[1]] += learning_rate * (error - reg * bi[k[1]])
    return bu, bi


def pred(para_m, para_test, para_mu, para_bu, para_bi):
    """
        predict on the test data
    :param para_m:      rating matrix
    :param para_test:   test_data
    :param para_mu:     the average of all score
    :param para_bu:     the bias of all user
    :param para_bi:     the bias of all item
    :return :           get the prediction score
    """
    res = []
    for i in range(len(para_test)):
        hat = para_mu + para_bu[para_test[i][0]] + para_bi[para_test[i][1]]
        res.append([para_test[i][0], para_test[i][1], hat]) 
    return res