import numpy as np
#
# import theano
# import theano.tensor as T
import keras
from keras import backend as K
from keras import initializers
from keras.regularizers import l1, l2, l1_l2
from keras.models import Sequential, Model
from keras.layers.core import Dense, Lambda, Activation
from keras.layers import Embedding, Input, Dense, merge, Reshape, Flatten, Dropout
from keras.optimizers import Adagrad, Adam, SGD, RMSprop
from evaluate_meta import evaluate_model
from Dataset import Dataset
from time import time
import sys
import argparse
import os
import pandas as pd
from keras.callbacks import TensorBoard


PATH_META_USER= "Data/u.vector"
PATH_META_ITEM="Data/i.vector"
#################### Arguments ####################
def parse_args():
    parser = argparse.ArgumentParser(description="Run NeuMF.")
    parser.add_argument('--path', nargs='?', default='Data/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='ml-1m',
                        help='Choose a dataset.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    parser.add_argument('--num_factors', type=int, default=8,
                        help='Embedding size of MF model.')
    parser.add_argument('--layers', nargs='?', default='[64,32,16,8]',
                        help="MLP layers. Note that the first layer is the concatenation of user and item embeddings. So layers[0]/2 is the embedding size.")
    parser.add_argument('--reg_mf', type=float, default=0,
                        help='Regularization for MF embeddings.')
    parser.add_argument('--reg_layers', nargs='?', default='[0,0,0,0]',
                        help="Regularization for each MLP layer. reg_layers[0] is the regularization for embeddings.")
    parser.add_argument('--num_neg', type=int, default=4,
                        help='Number of negative instances to pair with a positive instance.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--learner', nargs='?', default='adam',
                        help='Specify an optimizer: adagrad, adam, rmsprop, sgd')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Show performance per X iterations')
    parser.add_argument('--out', type=int, default=1,
                        help='Whether to save the trained model.')
    parser.add_argument('--mf_pretrain', nargs='?', default='',
                        help='Specify the pretrain model file for MF part. If empty, no pretrain will be used')
    parser.add_argument('--mlp_pretrain', nargs='?', default='',
                        help='Specify the pretrain model file for MLP part. If empty, no pretrain will be used')
    return parser.parse_args()

def get_model(num_users, num_items, mf_dim=10, layers=[10], reg_layers=[0], reg_mf=0):
    assert len(layers) == len(reg_layers)
    # assert (layers[0]%2) !=0
    num_layer = len(layers)  # Number of layers in the MLP
    # Input variables
    movie_input = keras.layers.Input(shape=[20,], name='Item',dtype=np.float32)
    movie_embedding_mlp = keras.layers.Embedding(num_items, int(layers[0] / 2), name='Movie-Embedding-MLP')(movie_input)
    movie_vec_mlp = keras.layers.Flatten(name='FlattenMovies-MLP')(movie_embedding_mlp)
    movie_vec_mlp = keras.layers.Dropout(rate=0.2)(movie_vec_mlp)

    movie_embedding_mf = keras.layers.Embedding(num_items, mf_dim, name='Movie-Embedding-MF')(movie_input)
    movie_vec_mf = keras.layers.Flatten(name='FlattenMovies-MF')(movie_embedding_mf)
    movie_vec_mf = keras.layers.Dropout(rate=0.2)(movie_vec_mf)

    user_input = keras.layers.Input(shape=[22,], name='User', dtype=np.float32)
    user_vec_mlp = keras.layers.Flatten(name='FlattenUsers-MLP')(
        keras.layers.Embedding(num_users, int(layers[0] / 2), name='User-Embedding-MLP')(user_input))
    user_vec_mlp = keras.layers.Dropout(rate=0.2)(user_vec_mlp)

    user_vec_mf = keras.layers.Flatten(name='FlattenUsers-MF')(
        keras.layers.Embedding(num_users, mf_dim, name='User-Embedding-MF')(user_input))
    user_vec_mf = keras.layers.Dropout(rate=0.2)(user_vec_mf)

    concat = keras.layers.merge.concatenate([movie_vec_mlp, user_vec_mlp], name="Concat")
    mlp_vector = keras.layers.Dropout(rate=0.2)(concat)
    for idx in range(1, num_layer):
        if (idx < 3):
            # mlp_vector = keras.layers.Dropout(0.2)(mlp_vector)
            layer_dense = keras.layers.Dense(layers[idx], activation='relu', name="layer%d" % idx)
            layer = keras.layers.BatchNormalization(name='Batch%d' % idx)
            mlp_vector = layer(layer_dense(mlp_vector))
        else:
            layer = keras.layers.Dense(layers[idx], activation='relu', name="layer%d" % idx)
            mlp_vector = layer(mlp_vector)

    pred_mlp = keras.layers.Dense(1, activation='relu', name='Activation')(mlp_vector)

    pred_mf = keras.layers.merge.concatenate([movie_vec_mf, user_vec_mf], name='Dot')

    combine_mlp_mf = keras.layers.merge.concatenate([pred_mf, pred_mlp], name='Concat-MF-MLP')
    # result_combine = keras.layers.Dense(100, name='Combine-MF-MLP')(combine_mlp_mf)
    # deep_combine = keras.layers.Dense(100, name='FullyConnected-4')(result_combine)

    result = keras.layers.Dense(1, activation="sigmoid", name='Prediction')(combine_mlp_mf)

    model = keras.Model([user_input, movie_input], result)
    return model

def load_pretrain_model(model, gmf_model, mlp_model, num_layers):
    # MF embeddings
    gmf_user_embeddings = gmf_model.get_layer('user_embedding').get_weights()
    gmf_item_embeddings = gmf_model.get_layer('item_embedding').get_weights()
    model.get_layer('mf_embedding_user').set_weights(gmf_user_embeddings)
    model.get_layer('mf_embedding_item').set_weights(gmf_item_embeddings)

    # MLP embeddings
    mlp_user_embeddings = mlp_model.get_layer('user_embedding').get_weights()
    mlp_item_embeddings = mlp_model.get_layer('item_embedding').get_weights()
    model.get_layer('mlp_embedding_user').set_weights(mlp_user_embeddings)
    model.get_layer('mlp_embedding_item').set_weights(mlp_item_embeddings)

    # MLP layers
    for i in range(1, num_layers):
        mlp_layer_weights = mlp_model.get_layer('layer%d' % i).get_weights()
        model.get_layer('layer%d' % i).set_weights(mlp_layer_weights)

    # Prediction weights
    gmf_prediction = gmf_model.get_layer('prediction').get_weights()
    mlp_prediction = mlp_model.get_layer('prediction').get_weights()
    new_weights = np.concatenate((gmf_prediction[0], mlp_prediction[0]), axis=0)
    new_b = gmf_prediction[1] + mlp_prediction[1]
    model.get_layer('prediction').set_weights([0.5 * new_weights, 0.5 * new_b])
    return model


def get_train_instances(train, num_negatives=100):
    print("\n on get instance")
    item_meta = pd.read_csv(PATH_META_ITEM, header=None, index_col=[0])
    user_meta = pd.read_csv(PATH_META_USER, header=None, index_col=[0])
    item_meta.columns = ['vector']
    user_meta.columns = ['vector']
    item_dict = item_meta.to_dict('index')
    user_dict = user_meta.to_dict('index')
    count_key = 0
    user_input, item_input, labels = [], [], []
    num_users = train.shape[0]

    for (u, i) in train.keys():
        # if (count_key % 10000 == 0 or count_key >= 994100):
            # print("\n-----------count_key: ", count_key)
            # positive instance
        u_meta = str(user_dict[u]['vector'])
        i_meta = str(item_dict[i]['vector'])
        user_input.append(np.fromstring(u_meta.strip('[').strip(']'), sep=',', dtype=np.float32))
        item_input.append(np.fromstring(i_meta.strip('[').strip(']'), sep=',', dtype=np.float32))
        labels.append(1)

        # negative instances
        count_nega = 0
        for t in range(num_negatives):
            # print("\n-----------t: ",t)
            if count_nega == num_negatives:
                break
            j = np.random.randint(num_items)
            while (u, j) in train:
                j = np.random.randint(num_items)
            try:
                j_meta = str(item_dict[j]['vector'])
            except:
                t -= 1
                continue

            user_input.append(np.fromstring(u_meta.strip('[').strip(']'), sep=',', dtype=np.float32))
            item_input.append(np.fromstring(j_meta.strip('[').strip(']'), sep=',', dtype=np.float32))
            labels.append(0)
            count_nega += 1
        count_key += 1
    return user_input, item_input, labels


if __name__ == '__main__':
    args = parse_args()
    num_epochs = args.epochs
    batch_size = args.batch_size
    mf_dim = args.num_factors
    layers = eval(args.layers)
    reg_mf = args.reg_mf
    reg_layers = eval(args.reg_layers)
    num_negatives = args.num_neg
    learning_rate = args.lr
    learner = args.learner
    verbose = args.verbose
    mf_pretrain = args.mf_pretrain
    mlp_pretrain = args.mlp_pretrain

    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    topK = 10
    evaluation_threads = 1  # mp.cpu_count()
    print("NeuMF arguments: %s " % (args))
    model_out_file = 'Pretrain/%s_NeuMF_%d_%s_%d.h5' % (args.dataset, mf_dim, args.layers, time())

    # Loading data
    t1 = time()
    # print(args.path)
    dataset = Dataset(args.path + args.dataset)
    train, testRatings, testNegatives = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives
    num_users, num_items = train.shape
    print("Load data done [%.1f s]. #user=%d, #item=%d, #train=%d, #test=%d"
          % (time() - t1, num_users, num_items, train.nnz, len(testRatings)))

    # Build model
    model = get_model(num_users, num_items, mf_dim, layers, reg_layers, reg_mf)
    if learner.lower() == "adagrad":
        model.compile(optimizer=Adagrad(lr=learning_rate), loss='binary_crossentropy')
    elif learner.lower() == "rmsprop":
        model.compile(optimizer=RMSprop(lr=learning_rate), loss='binary_crossentropy')
    elif learner.lower() == "adam":
        model.compile(optimizer=Adam(lr=learning_rate), loss='binary_crossentropy')
    else:
        model.compile(optimizer=SGD(lr=learning_rate), loss='binary_crossentropy')

    # Load pretrain model
    # if mf_pretrain != '' and mlp_pretrain != '':
    #     gmf_model = GMF.get_model(num_users, num_items, mf_dim)
    #     gmf_model.load_weights(mf_pretrain)
    #     mlp_model = MLP.get_model(num_users, num_items, layers, reg_layers)
    #     mlp_model.load_weights(mlp_pretrain)
    #     model = load_pretrain_model(model, gmf_model, mlp_model, len(layers))
    #     print("Load pretrained GMF (%s) and MLP (%s) models done. " % (mf_pretrain, mlp_pretrain))
    # file_writer = tf.summary.create_file_writer(logdir + "/metrics")
    # file_writer.set_as_default()

    logs = "logs/{}".format(time())
    tensorBoard = TensorBoard(log_dir="logs")

    ls_HR=[]
    ls_NDCG=[]
    ls_loss= []
    # Init performance
    (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, topK, evaluation_threads)
    hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
    print('Init: HR = %.4f, NDCG = %.4f' % (hr, ndcg))
    ls_HR.append(hr)
    ls_NDCG.append(ndcg)

    best_hr, best_ndcg, best_iter = hr, ndcg, -1
    if args.out > 0:
        model.save_weights(model_out_file, overwrite=True)
    print("\nbefore training\n")
        # Training model
    for epoch in range(num_epochs):
        t1 = time()
        print("\non training\n")
        # Generate training instances
        user_input, item_input, labels = get_train_instances(train, num_negatives)
        print("in training epoch: ", epoch)
        # Training
        hist = model.fit([np.array(user_input), np.array(item_input)],  # input
                         np.array(labels),  # labels
                         batch_size=batch_size, epochs=1, verbose=1, shuffle=True, callbacks=[tensorBoard])
        t2 = time()

        # Evaluation
        if epoch % verbose == 0:
            (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, topK, evaluation_threads)
            hr, ndcg, loss = np.array(hits).mean(), np.array(ndcgs).mean(), hist.history['loss'][0]
            ls_loss.append(hr)
            ls_HR.append(hr)
            ls_NDCG.append(ndcg)
            print('Iteration %d [%.1f s]: HR = %.4f, NDCG = %.4f, loss = %.4f [%.1f s]'
                  % (epoch, t2 - t1, hr, ndcg, loss, time() - t2))
            if hr > best_hr:
                best_hr, best_ndcg, best_iter = hr, ndcg, epoch
                if args.out > 0:
                    model.save_weights(model_out_file, overwrite=True)

    history_training = open("./his_{}_{}_{}.txt".format(learning_rate, layers, mf_dim), "w")
    history_training.write("HR:{}".format(ls_HR))
    history_training.write("NDCG:{}".format(ls_NDCG))
    history_training.write("LOSS:{}".format(ls_loss))
    history_training.close()

    print("End. Best Iteration %d:  HR = %.4f, NDCG = %.4f. " % (best_iter, best_hr, best_ndcg))
    if args.out > 0:
        print("The best NeuMF model is saved to %s" % (model_out_file))
