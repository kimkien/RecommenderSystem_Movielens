import numpy as np
import scipy.sparse as sp
import pandas as pd


def load_rating_file_as_list(filename):
    ratingList = []
    with open(filename, "r") as f:
        line = f.readline()
        while line != None and line != "":
            arr = line.split("\t")
            user, item = int(arr[0]), int(arr[1])
            ratingList.append([user, item])
            line = f.readline()
    return ratingList

def create_negative_file(path_file= "Data/ml-1m/1m_train.csv", num_samples=100):
    # Get number of users and items
    num_users, num_items = 0, 0
    with open(path_file, "r") as f:
        line = f.readline()
        while line != None and line != "":
            arr = line.split("\t")
            # print("arr[0]",     arr[0])
            u, i = int(arr[0]), int(arr[1])
            num_users = max(num_users, u)
            num_items = max(num_items, i)
            line = f.readline()
    trainMatrix = sp.dok_matrix((num_users + 1, num_items + 1), dtype=np.float32)
    with open(path_file, "r") as f:
        line = f.readline()
        while line != None and line != "":
            arr = line.split("\t")
            user, item, rating = int(arr[0]), int(arr[1]), float(arr[2])
            if (rating > 0):
                trainMatrix[user, item] = 1.0
            line = f.readline()

    negativeList = []
    testRatings= load_rating_file_as_list("Data/ml-1m/1m_test.csv")
    for user_item_pair in testRatings:
        user = user_item_pair[0]
        item = user_item_pair[1]
        negatives = [(user,item)]
        for t in range(num_samples):
            j = np.random.randint(1,  num_items)
            while (user, j) in  trainMatrix or j == item:
                j = np.random.randint(1,  num_items)
            negatives.append(j)
        negativeList.append(negatives)
    df= pd.DataFrame(data=negativeList)
    print(df.head(2))
    df.to_csv("./Data/1m_negative.csv", header=False, index=False, sep="\t")
    return negativeList

if __name__ == '__main__':
    # ------------CREATE TEST NEGATIVE
    # create_negative_file()
    # ------------CREATE TRAIN NEGATIVE
    create_negative_file(4)