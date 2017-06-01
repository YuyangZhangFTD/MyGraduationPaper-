"""
    Mine paper
"""
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as la 
from tensorflow.python.framework import ops
import pandas as pd
from scipy import sparse
import RecTool as rt
import item_feature as itf
from collections import OrderedDict
from sklearn.manifold import TSNE


def count_dict(para_item_dict, count_th=3):
    """
        count and get the word dictionary
    onehot_dict = {
        0:              [unfrequent item list]
        item_index+1:   one hot encoding index 
    }
    :param para_item_dict:  item dictionary
    :param count_th:        count threshold, default 3      
    :return:                the one hot coding dictionary
    """
    # 0 for unfrequent item
    onehot_dict = {0:[]}
    key_max = 1
    for k,v in para_item_dict.items():
        # above 3 times for a item
        if len(v) > count_th:
            onehot_dict[k+1] = key_max
            key_max += 1
        else:
            onehot_dict[0].append(k)
    # onehot_matrix = np.mat(np.diag(np.ones(key_max)))
    return onehot_dict#, onehot_matrix


def lookup_dic(para_onehot_dict, para_index):
    """
        the function for search one hot coding index by item index
    :param para_onehot_dict:    the one hot codint dictionary
    :param para_index:          the need-searching item index
    :return:                    the one hot coding index
    """
    try:
        return para_onehot_dict[para_index + 1]
    except:
        return 0


def get_onehot(para_onehot_dict, para_onehot_mat, para_index):
    """
        to get one hot coding vector by item index
    :param para_onehot_dict:    the one hot codint dictionary
    :param para_onehot_mat:     the one hot matrix(a identity matrix)
    :param para_index:          the need-searching item index
    :return:                    the one hot coding index    
    """
    try:
        return para_onehot_mat[para_onehot_dict[para_index + 1],:].T
    except:
        return para_onehot_mat[0,:].T


def skip_gram(para_user_dict, batch_size, para_onehot_dict, window_size=2):
    """
        the SkipGram model to generate data
    :param para_user_dict:      the user dictionary
    :param batch_size:          the batch size of trian data
    :param para_onehot_dict:    the one hot codint dictionary
    :param window_size:         the window size, default 2
    :return:                    batch data and label data
    """
    batch_data = []
    label_data = []
    dic = para_onehot_dict
    while len(batch_data) < batch_size:
        rand_user = np.random.choice(len(para_user_dict.keys()))
        user_list = para_user_dict[rand_user] 

        user_list.sort(key=lambda tple: tple[1])
        batch_and_labels = []
        max_len = len(user_list)
        for i in range(window_size,max_len-window_size):
            batch_and_labels += [
                (
                    lookup_dic(para_onehot_dict, user_list[i][0]), 
                    lookup_dic(para_onehot_dict, user_list[i-ws][0])
                ) 
                for ws in range(1,window_size+1)
            ]   
            batch_and_labels += [
                (
                    lookup_dic(para_onehot_dict, user_list[i][0]), 
                    lookup_dic(para_onehot_dict, user_list[i+ws][0])
                )
                for ws in range(1,window_size+1)
            ]

        batch = [x[0] for x in batch_and_labels]
        labels = [x[1] for x in batch_and_labels]
        batch_data.extend(batch[:batch_size])
        label_data.extend(labels[:batch_size])

    batch_data = batch_data[:batch_size]
    label_data = label_data[:batch_size]
    batch_data = np.array(batch_data)
    label_data = np.transpose(np.array([label_data]))
    return batch_data, label_data


def generate_batch_data(para_user_dict, batch_size, para_onehot_dict, window_size=2, method='skip_gram'):
    """
        to generate data
    :param para_user_dict:      the user dictionary
    :param batch_size:          the batch size of trian data
    :param para_onehot_dict:    the one hot codint dictionary
    :param window_size:         the window size, default 2
    :param method:              the generate method, default skip_gram
    :return:                    batch data and label data   
    """
    if method=='skip_gram':
        return skip_gram(para_user_dict, batch_size, para_onehot_dict, window_size=window_size)
    elif method=='CBOW':
        return CBOW(para_user_dict, batch_size, para_onehot_dict, window_size=window_size)
    else:
        print("Wrong method.")
        return None


def train(para_m, para_user_dict, para_item_dict, embedding_size=500, iter_n=50000,\
            window_size=3, batch_size=100, count_th=3, plot_loss=True, method='skip_gram'): 
    """
        trian model
    :param para_m:              the rating matrix
    :param para_user_dict:      the user dictionary
    :param para_item_dict:      the item dictionary
    :param embedding_size:      the embedding size, default 500
    :param iter_n:              the iteration number, default 50000               
    :param batch_size:          the batch size of trian data, default 100
    :param window_size:         the window size, default 3
    :param count_th:            the count threshold, default 3
    :param plot_loss:           whether to plot loss, default True
    :param method:              the generate method, default skip_gram
    :return:                    the embedding matrix, loss list([x,y]), one hot coding dictionary 
    
    """
    num_sampled=int(batch_size/2)  
    onehot_dict = count_dict(para_item_dict, count_th=count_th)
    sess = tf.Session()
    vocabulary_size = len(onehot_dict.keys())

    embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))

    nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size],
                                                stddev=1.0 / np.sqrt(embedding_size)))
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))
    x_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    y_target = tf.placeholder(tf.int32, shape=[batch_size, 1])
    embed = tf.nn.embedding_lookup(embeddings, x_inputs)
    loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights,
                                        biases=nce_biases,
                                        labels=y_target,
                                        inputs=embed,
                                        num_sampled=num_sampled,
                                        num_classes=vocabulary_size))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0).minimize(loss)
    init = tf.global_variables_initializer()
    sess.run(init)
    loss_vec = []
    loss_x_vec = []
    for iter_i in range(iter_n):
        batch_inputs, batch_labels = generate_batch_data(para_user_dict, batch_size,\
                                 onehot_dict, window_size, method)
        feed_dict = {x_inputs : batch_inputs, y_target : batch_labels}
        sess.run(optimizer, feed_dict=feed_dict)
        if (iter_i+1) % 100 == 0:
            loss_val = sess.run(loss, feed_dict=feed_dict)
            loss_vec.append(loss_val)
            loss_x_vec.append(iter_i+1)
            # print("Loss at step {} : {}".format(iter_i+1, loss_val))
    embedding_mat = sess.run(embeddings)
    sess.close()

    if plot_loss:
        import matplotlib.pyplot as plt
        plt.plot(loss_x_vec, loss_vec)
        plt.show()
    
    return embedding_mat, [loss_x_vec, loss_vec], onehot_dict


def sim_cos(v1, v2):
    """
        compute similarity between vector1 and vector2
    :param v1:  vector 1
    :param v2:  vecotr 2
    :return:    similarity between v1 and v2
    """
    num = v1.T*v2
    den = la.norm(v1)*la.norm(v2) 
    return num/den

# what a smart boy I am.
def compute_sim_cos(para_onehot_dict, para_embed_mat):
    """
        compute similarity by matrix computation
    :param para_onehot_dict:    one hot coding dictionary
    :param para_embed_mat:      the embedding matrix
    :return:                    the similarity matrix
    """
    item_list = list(para_onehot_dict.keys())
    item_n = len(item_list)
    sim = np.zeros([item_n, item_n])
    onehot_mat = np.mat(np.diag(np.ones(item_n)))
    embed = onehot_mat*para_embed_mat
    num = embed*embed.T
    den_sq = la.norm(embed, axis=1)
    for i in range(1,item_n):
        for j in range(1, item_n):
            if j > i:
                sim[i,j] = num[i,j]/(den_sq[i]*den_sq[j]) 
            elif i == j:
                sim[i,j] = 0
            else:
                sim[i,j] = sim[j,i]
        # print("Finish:  "+str(i)+"/"+str(item_n))
    return sim


def compute_sim(para_onehot_dict, para_embed_mat, para_sim=sim_cos):
    """
        old way to compute similarity matrix
    """
    item_list = list(para_onehot_dict.keys())
    item_n = len(item_list)
    sim = np.zeros([item_n, item_n])
    onehot_mat = np.mat(np.diag(np.ones(item_n)))
    for i in range(1,item_n):
        for j in range(1,item_n):
            if j > i:
                v1 = para_embed_mat.T*onehot_mat[:,i]
                v2 = para_embed_mat.T*onehot_mat[:,j]
                sim[i,j] = para_sim(v1,v2)
        print("Finish:  "+str(i)+"/"+str(item_n))

    return sim

def pred(para_m, para_test, para_embed_mat, para_onehot_dict, para_user_dict, para_n_sim=5):
    """
        to predict rating in test data
    :param para_m:              the rating matrix
    :param para_test:           the test data
    :param para_embed_mat:      the embedding matrix
    :param para_onehot_dict:    the one hot coding dictionary
    :param para_user_dict:      the user dictionary
    :param para_n_sim:          the similar item to be considered
    :return:                    the estimated result
    """
    item_list = list(para_onehot_dict.keys())
    item_n = len(item_list)
    onehot_mat = np.mat(np.diag(np.ones(item_n)))
    sim_matrix = compute_sim_cos(para_onehot_dict, para_embed_mat)
    count_skip = 0
    res = []
    for test in para_test:
        user = test[0]
        item = test[1]
        try:
            embed_index = para_onehot_dict[item+1]
        except:
            count_skip += 1
            res.append(test)
            continue
        
        try:
            neighbors = [
                (i, sim_matrix[embed_index, para_onehot_dict[i+1]], r) 
                for (i, r) in para_user_dict[user]
                if i+1 in item_list
            ]
            r_mean = np.mean([r for (x,r) in para_user_dict[user]])
            neighbors = sorted(neighbors, key=lambda tple: tple[1], reverse=True)
            sum_sim = sum_ratings = actual_k = 0
            for (_, sim, r) in neighbors[:para_n_sim]:  # n>len is ok
                if sim > 0:
                    sum_sim += sim
                    sum_ratings += sim * (r-r_mean)
                    actual_k += 1
            if actual_k < para_n_sim:
                print("Not enough similar users!")
                print("User: "+str(user)+"  Item: "+str(item))       
            res.append([user, item, sum_ratings/sum_sim+r_mean])
        except:
            print("Something Wrong!!")
            print("User:  "+str(user)+"  Item:  "+str(item))
            count_skip += 1
            res.append(test)

    print("Skip:  "+str(count_skip))
    return res, count_skip



   