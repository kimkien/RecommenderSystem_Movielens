'''
Created on Apr 15, 2016
Evaluate the performance of Top-K recommendation:
    Protocol: leave-1-out evaluation
    Measures: Hit Ratio and NDCG
    (more details are in: Xiangnan He, et al. Fast Matrix Factorization for Online Recommendation with Implicit Feedback. SIGIR'16)

@author: hexiangnan
'''
import math
import heapq  # for retrieval topK
import multiprocessing
import numpy as np
from time import time
import pandas as pd

# from numba import jit, autojit

# Global variables that are shared across processes
_model = None
_testRatings = None
_testNegatives = None
_K = None
PATH_META_USER = "Data/u.vector"
PATH_META_ITEM = "Data/i.vector"


def evaluate_model(model, testRatings, testNegatives, K, num_thread):
    """
    Evaluate the performance (Hit_Ratio, NDCG) of top-K recommendation
    Return: score of each test rating.
    """
    global _model
    global _testRatings
    global _testNegatives
    global _K
    _model = model
    _testRatings = testRatings
    _testNegatives = testNegatives
    _K = K

    hits, ndcgs = [], []
    if (num_thread > 1):  # Multi-thread
        pool = multiprocessing.Pool(processes=num_thread)
        res = pool.map(eval_one_rating, range(len(_testRatings)))
        pool.close()
        pool.join()
        hits = [r[0] for r in res]
        ndcgs = [r[1] for r in res]
        return (hits, ndcgs)
    # Single thread
    for idx in range(len(_testRatings)):
        if (idx%1000==0):
            print("\nEvaluate_meta-----", idx)
        (hr, ndcg) = eval_one_rating(idx)
        hits.append(hr)
        ndcgs.append(ndcg)
    return (hits, ndcgs)


def eval_one_rating(idx):
    rating = _testRatings[idx]
    items = _testNegatives[idx]
    u = rating[0]
    gtItem = rating[1]
    items.append(gtItem)
    # get instance of input
    user_meta = pd.read_csv(PATH_META_USER,header=None, index_col=[0])
    item_meta = pd.read_csv(PATH_META_ITEM,header=None, index_col=[0])
    item_meta.columns = ['vector']
    user_meta.columns = ['vector']
    item_dict = item_meta.to_dict('index')
    user_dict = user_meta.to_dict('index')

    # if(u==6038):
    #     print(user_meta)
    ls_u_input = []
    u_temp = str(user_dict[u]['vector'])

    for i in range(len(items)):
        ls_u_input.append(np.fromstring(u_temp.strip('[').strip(']'), sep=',', dtype=np.float32))

    ls_i_input = []
    for item in items:
        temp= str(item_dict[i]['vector'])
        ls_i_input.append(np.fromstring(temp.strip('[').strip(']'), sep=',', dtype= np.float32))
    # Get prediction scores
    map_item_score = {}
    arr_u= np.array((ls_u_input), dtype=np.float32)
    arr_i= np.array((ls_i_input), dtype=np.float32)
    predictions = _model.predict([arr_u, arr_i], verbose=0)
    for i in range(len(items)):
        item = items[i]
        map_item_score[item] = predictions[i]
    items.pop()

    # Evaluate top rank list
    ranklist = heapq.nlargest(_K, map_item_score, key=map_item_score.get)
    hr = getHitRatio(ranklist, gtItem)
    ndcg = getNDCG(ranklist, gtItem)
    return (hr, ndcg)


def getHitRatio(ranklist, gtItem):
    for item in ranklist:
        if item == gtItem:
            return 1
    return 0

def getNDCG(ranklist, gtItem):
    for i in range(len(ranklist)):
        item = ranklist[i]
        if item == gtItem:
            return math.log(2) / math.log(i + 2)
    return 0

