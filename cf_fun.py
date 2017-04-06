"""
    Functions in Collaborative Filter 
"""
import numpy as np
import RecTool as rt


def get_common(para_m, para_p1, para_p2):
    """
        Get common item between para_p1 and para_p2
    :param para_m:      rating matrix
    :param para_p1:     user 1
    :param para_p2:     user 2
    :return :           a list of common item of 2 users
    """
    p1_list = []
    p2_list = []
    for k in para_m[para_p1].keys():
        p1_list.append(k[1])
    for k in para_m[para_p2].keys():
        p2_list.append(k[1])
    com_list = []
    for item in p1_list:
        if item in p2_list:
            com_list.append(item)
    return com_list


def sim_distance(para_m, para_p1, para_p2):
    """
        compute similarity by distance
    :param para_m:      rating matrix
    :param para_p1:     user 1
    :param para_p2:     user 2
    :return :           similarity(scalar)
    """
    com_list = get_common(para_m, para_p1, para_p2)
    if len(com_list) == 0:
        return 0.0      
    sum_square = sum([(para_m[para_p1, x]-para_m[para_p2, x])**2 for x in com_list])
    return 1/(1 + np.sqrt(sum_square))


def sim_pearson(para_m, para_p1, para_p2):
    """
        compute similarity by pearson coefficient
    :param para_m:      rating matrix
    :param para_p1:     user 1
    :param para_p2:     user 2
    :return :           similarity(scalar)
    """
    com_list = get_common(para_m, para_p1, para_p2)
    if len(com_list) == 0:
        return 0.0    
    n = len(com_list)
    sum1 = sum([para_m[para_p1, x] for x in com_list])
    sum2 = sum([para_m[para_p2, x] for x in com_list])
    sum1Sq = sum([pow(para_m[para_p1, x],2) for x in com_list])
    sum2Sq = sum([pow(para_m[para_p2, x],2) for x in com_list])
    pSum = sum([para_m[para_p1, x]*para_m[para_p2, x] for x in com_list])
    num = pSum - (sum1 * sum2 / n)
    den = np.sqrt((sum1Sq-pow(sum1,2)/n) * (sum2Sq-pow(sum2,2)/n))
    if den==0: 
        return 0
    r=num/den
    return r


def sim_cos(para_m, para_p1, para_p2):
    """
        compute similarity by cosin distance
    :param para_m:      rating matrix
    :param para_p1:     user 1
    :param para_p2:     user 2
    :return :           similarity(scalar)
    """
    com_list = get_common(para_m, para_p1, para_p2)
    if len(com_list) == 0:
        return 0.0    
    vec1 = [para_m[para_p1, x] for x in com_list]
    vec2 = [para_m[para_p2, x] for x in com_list]
    num = sum([para_m[para_p1, x] * para_m[para_p2, x] for x in com_list])
    den = np.sqrt(sum([para_m[para_p1, x]**2 for x in com_list])) \
            * np.sqrt(sum([para_m[para_p2, x]**2 for x in com_list]))
    return num/den




def compute_sim(para_m, para_n_sim=50, para_sim=sim_pearson, user_based=True):
    """
        Compute similarity matrix
    :param para_m:      rating matrix
    :param para_n_sim:  the number of similar users
    :param para_sim:    the function that measures the similarity
    :return :           the similarity matrix
    """
    
    return None


def pred(para_m, para_test, para_n_sim=50, para_sim=sim_pearson, user_based=True):
    """
        predict on the test data
    :param para_m:      rating matrix
    :param para_test:   test_data
    :param para_n_sim:  the number of similar users
    :param para_sim:    the function that measures the similarity
    :return:            get the prediction
    """
    sim_matrix = compute_sim(para_m, para_n_sim=para_n_sim, para_sim=para_sim, user_based=user_based)
    
    return None




# =========================== Maybe not useful things =====================================
def recommend(para_m, para_p, para_n_sim=50, para_n_rec=50, para_sim=sim_pearson):
    sim_u = get_top(para_m, para_p, para_n=para_n_sim, para_sim=sim_pearson)
    # TODO
    return None


def get_top(para_m, para_p1, para_n=50, para_sim=sim_pearson):
    """
        get top similar user
    :param para_m:      rating matrix
    :param para_p1:     user 1
    :param para_p2:     user 2
    :param para_n_sim:  the number of similar users
    :param para_sim:    the function that measures the similarity  
    :return :           a list record the most similar users  
    """
    num_user, num_item = para_m.shape
    scores=[(para_sim(para_m, para_p1, x), x) for x in range(num_user) if x != para_p1] 
    scores.sort()
    scores.reverse()
    return [x[1] for x in scores[0:para_n]]


def pred_one(para_m, para_p, para_i, para_n_sim=50, para_sim=sim_pearson):
    """
        predict the score of item i rated by user i based on similar users
    :param para_m:      rating matrix
    :param para_p:      user
    :param para_i:      item
    :param para_n_sim:  the number of similar users
    :param para_sim:    the function that measures the similarity
    :return :           get the prediction
    """
    sim_u = get_top(para_m, para_p, para_n=para_n_sim, para_sim=sim_pearson)
    com_list = [x for x in sim_u if para_m[x, para_i] != 0]
    if len(com_list) == 0:
        # with no common item, calculate the average score
        return np.sum(para_m[:, para_i])/para_m[:,para_i].getnnz()
    else:
        return sum([para_m[x, para_i] for x in com_list]) / len(com_list)
