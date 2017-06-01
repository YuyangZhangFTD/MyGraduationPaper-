"""
    Functions in Collaborative Filter 
"""
import numpy as np
import RecTool as rt


def get_common(para_m, para_x_dict, para_1, para_2):
    """
        Get common item between para_1 and para_2
    :param para_m:      rating matrix
    :param para_x_dict: dict record by user or item
    :param para_1:      user 1 or item 1
    :param para_2:      user 2 or item 2
    :return :           a list of common things.
    """
    list_1 = para_x_dict[para_1]
    list_2 = para_x_dict[para_2]
    com_list = []
    for k1, __ in list_1:
        for k2, __ in list_2:
            if k1 == k2:            # instead of using dict here
                com_list.append(k1)
    return com_list


def sim_distance(para_m, para_x_dict, para_1, para_2, user_based=True):
    """
     compute similarity by distance
    :param para_m:      rating matrix
    :param para_x_dict: dict record by user or item
    :param para_1:      user 1 or item 1
    :param para_2:      user 2 or item 2
    :param user_based:  based on user or item
    :return :           similarity(scalar)
    """
    com_list = get_common(para_m, para_x_dict, para_1, para_2)
    if len(com_list) == 0:
        return 0.0      
    if user_based:
        sum_square = sum([(para_m[para_1, x]-para_m[para_2, x])**2 for x in com_list])
    else:
        sum_square = sum([(para_m[x, para_1]-para_m[x, para_2])**2 for x in com_list])
    return 1/(1 + np.sqrt(sum_square))


def sim_pearson(para_m, para_x_dict, para_1, para_2, user_based=True):
    """
        compute similarity by pearson coefficient
    :param para_m:      rating matrix
    :param para_x_dict: dict record by user or item    
    :param para_1:      user 1 or item 1
    :param para_2:      user 2 or item 2
    :param user_based:  based on user or item
    :return :           similarity(scalar)
    """
    com_list = get_common(para_m, para_x_dict, para_1, para_2)
    if len(com_list) == 0:
        return 0.0
    n = len(com_list)
    if user_based:
        sum1 = sum([para_m[para_1, x] for x in com_list])
        sum2 = sum([para_m[para_2, x] for x in com_list])
        sum1Sq = sum([pow(para_m[para_1, x],2) for x in com_list])
        sum2Sq = sum([pow(para_m[para_2, x],2) for x in com_list])
        pSum = sum([para_m[para_1, x]*para_m[para_2, x] for x in com_list])
    else:
        sum1 = sum([para_m[x, para_1] for x in com_list])
        sum2 = sum([para_m[x, para_2] for x in com_list])
        sum1Sq = sum([pow(para_m[x, para_1],2) for x in com_list])
        sum2Sq = sum([pow(para_m[x, para_2],2) for x in com_list])
        pSum = sum([para_m[x, para_1]*para_m[x, para_2] for x in com_list])
    num = pSum - (sum1 * sum2 / n)
    den = np.sqrt((sum1Sq-pow(sum1,2)/n) * (sum2Sq-pow(sum2,2)/n))
    if den==0: 
        return 0
    return num/den


def sim_cos(para_m, para_x_dict, para_1, para_2, user_based=True):
    """
        compute similarity by cosin distance
    :param para_m:      rating matrix
    :param para_x_dict: dict record by user or item
    :param para_1:      user 1 or item 1
    :param para_2:      user 2 or item 2
    :param user_based:  based on user or item
    :return :           similarity(scalar)
    """
    com_list = get_common(para_m, para_x_dict, para_1, para_2)
    if len(com_list) == 0:
        return 0.0
    if user_based:
        vec1 = [para_m[para_1, x] for x in com_list]
        vec2 = [para_m[para_2, x] for x in com_list]
        num = sum([para_m[para_1, x] * para_m[para_2, x] for x in com_list])
        den = np.sqrt(sum([para_m[para_1, x]**2 for x in com_list])) \
            * np.sqrt(sum([para_m[para_2, x]**2 for x in com_list]))
    else:
        vec1 = [para_m[x, para_1] for x in com_list]
        vec2 = [para_m[x, para_2] for x in com_list]
        num = sum([para_m[x, para_1] * para_m[x, para_2] for x in com_list])
        den = np.sqrt(sum([para_m[x, para_2]**2 for x in com_list])) \
            * np.sqrt(sum([para_m[x, para_2]**2 for x in com_list]))
    return num/den


@rt.fn_timer
def compute_sim(para_m, para_user_dict, para_item_dict, para_sim=sim_pearson, user_based=True):
    """
        Compute similarity matrix
    :param para_m:      rating matrix
    :param para_sim:    the function that measures the similarity
    :return :           the similarity matrix
    """
    num_user, num_item = para_m.shape
    if user_based:
        sim = np.zeros([num_user, num_user])
        x_dict = para_user_dict
    else:
        x_dict = para_item_dict
        sim = np.zeros([num_item, num_item])
    key_list = x_dict.keys()
    for k1 in key_list:
        for k2 in key_list:
            if k1 == k2:
                sim[k1,k2] = 1
            else:
                if sim[k2,k1] != 0:
                    sim[k1,k2] = sim[k2,k1]
                else:
                    # calculate similarity
                    sim[k1, k2] = para_sim(para_m, x_dict, k1, k2, user_based=user_based)
    return sim


@rt.fn_timer
def pred(para_m, para_test, para_user_dict, para_item_dict, \
            para_n_sim=3, para_sim=sim_pearson, user_based=True):
    """
        predict on the test data
    :param para_m:      rating matrix
    :param para_test:   test_data
    :param para_n_sim:  the number of similar users
    :param para_sim:    the function that measures the similarity
    :return:            get the prediction
    """
    sim_matrix = compute_sim(para_m, para_user_dict, para_item_dict, \
                                para_sim=para_sim, user_based=user_based)
    res = []
    random_count = 0

    if user_based:
        for test in para_test:
            user = test[0]
            item = test[1]
            try:
                neighbors = [(x, sim_matrix[user, x], r) for (x, r) in para_item_dict[item]]
                neighbors = sorted(neighbors, key=lambda tple: tple[1], reverse=True)
                sum_sim = sum_ratings = actual_k = 0
                for (_, sim, r) in neighbors[:para_n_sim]:
                    if sim > 0:
                        sum_sim += sim
                        sum_ratings += sim * r
                        actual_k += 1
                if actual_k < para_n_sim:
                    print("Not enough similar users!")
                    print("User: "+str(user)+"  Item: "+str(item))       
                res.append([user, item, sum_ratings/sum_sim])
            except:
                print("Random rating")
                print("User: "+str(user)+"  Item: "+str(item))  
                random_count += 1
                res.append([user, item, np.random.uniform(1,5)])
    else:
        for test in para_test:
            user = test[0]
            item = test[1]
            try:
                neighbors = [(x, sim_matrix[item, x], r) for (x, r) in para_user_dict[user]]
                neighbors = sorted(neighbors, key=lambda tple: tple[1], reverse=True)
                sum_sim = sum_ratings = actual_k = 0
                for (_, sim, r) in neighbors[:para_n_sim]:
                    if sim > 0:
                        sum_sim += sim
                        sum_ratings += sim * r
                        actual_k += 1
                if actual_k < para_n_sim:
                    print("Not enough similar users!")
                    print("User: "+str(user)+"  Item: "+str(item))       
                res.append([user, item, sum_ratings/sum_sim])
            except:
                print("Random rating")
                print("User: "+str(user)+"  Item: "+str(item))  
                random_count += 1
                res.append([user, item, np.random.uniform(1,5)])
    print("Random count:  "+str(random_count))
    return res

