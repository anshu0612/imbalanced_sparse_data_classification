import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras.backend as K
from keras.preprocessing import sequence
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional, Input, BatchNormalization, Conv1D, Conv2D, \
    Multiply, Activation, MaxPooling1D
from keras.regularizers import l2
from keras.optimizers import Adam, SGD
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score
import lightgbm as gbm
from sklearn.ensemble import RandomForestRegressor
## Below path hardcoded. TODO: Change this
prefix_path = 'data1'

labels = pd.read_csv(prefix_path + '/train_kaggle.csv')

print('Labels', labels.describe())

iterations = 6

test_X = []

# sparse_index = [0, 1, 4, 6, 8, 9, 10, 14, 16, 19, 21, 22, 23, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 38]
sparse_index = [i for i in range(40)]
dense_index = [2, 3, 5, 7, 11, 12, 13, 15, 17, 18, 20, 24, 29, 33, 35, 37, 39]


def __preprocess_feature(feat):
    sparse_x = feat[:, sparse_index]
    # fl = [0, np.array(sparse_x).shape[0] - 1]
    # sparse_x = sparse_x[fl, :].reshape((np.array(sparse_x).shape[1] * 2, 1))
    # spx = sparse_x[0, :].extend(sparse_x[np.array(sparse_x).shape[0] - 1, :])
    dense_x = feat[:, dense_index]
    return sparse_x, dense_x


for fileno in range(10000):
    ## zeros_array used to keep the maximum number of sequences constant to max_len
    # zeros_array = np.zeros((max_len, 40))

    ## features is a (N, 40) matrix
    features = np.load(prefix_path + '/test/test/' + str(fileno) + '.npy')

    sparse_x, dense_x = __preprocess_feature(np.array(features))

    ## For each feature, we find average of all values and replace all NaN with that value
    sparse_means = np.nanmean(np.where(sparse_x != 0, sparse_x, np.nan), axis=0)
    dense_means = np.nanmean(np.where(sparse_x != 0, sparse_x, np.nan), axis=0)
    test_X.append(sparse_means)

test_set_results = []

for it in range(10):
    print('Starting XGBoost Iteration ', it)
    X = []
    y = []
    ## ones count kept to balance number of zeros and ones in data to be equal
    ones = len(labels.loc[labels['label'] == 1])
    shuffled_labels = shuffle(labels)
    shuffled_y = np.array(shuffled_labels['label'])
    ## For each sample in the file
    X_sparse = []
    zero_test = []
    zero_test_y = []
    for index, train_label in shuffled_labels.iterrows():
        label = train_label['label']
        # Checking below if number of zeros matches total number of ones, then stop adding zeros to data
        if label == 0 and ones > 0:
            ones = ones - 0.85
        if ones <= 0 and label == 0:
            sparse = __preprocess_feature(features)[0]
            zero_test.append(np.nanmean(np.where(sparse != 0, sparse, np.nan), axis=0))
            zero_test_y.append(0)
            continue
        ## features is a (N, 40) matrix
        features = np.load(prefix_path + '/train/train/' + str(train_label['Id']) + '.npy')

        sparse_x, dense_x = __preprocess_feature(features)
        # For each feature, we find average of all values and replace all NaN with that value
        # for feature in range(40):
        #    average_value = np.average(features[:feature][np.nan_to_num(features[:feature]) != 0])
        #    features[:feature] = np.nan_to_num(features[:feature], average_value)

        feature_avg = np.average(features, axis=0)
        # assert feature_avg.shape == (features.shape[1],)

        sparse_means = np.nanmean(np.where(sparse_x != 0, sparse_x, np.nan), axis=0)
        # u, l = np.percentile(sparse_means, [1, 99])
        # y = np.clip(sparse_means, u, l)

        X_sparse.append(sparse_means)
        y.append(label)

    X_sparse = np.nan_to_num(np.array(X_sparse))
    y = np.array(y)
    ## Split into train and test datasets
    x_train, x_test, y_train, y_test = train_test_split(X_sparse, y, test_size=0.20)

    x_test_2 = np.concatenate([x_test])
    y_test_2 = np.concatenate([y_test])

    print('Starting XGB training')
    # lgb_train = gbm.Dataset(x_train, y_train)
    # lgb_eval = gbm.Dataset(x_test, y_test, reference=lgb_train)
    # params = {
    #     'boosting_type': 'dart',
    #     'objective': 'binary',
    #     'metric': {'l2', 'l1'},
    #     # 'max_bin': ,
    #     'num_leaves': 128,
    #     # 'max_depth': ,
    #     'learning_rate': 0.005,
    #     'feature_fraction': 0.9,
    #     'bagging_fraction': 0.8,
    #     'bagging_freq': 10,
    #     'verbose': 0,
    #     "tree_learner": "feature"
    # }

    ################
    # from sklearn.tree import DecisionTreeRegressor
    # from sklearn.ensemble import GradientBoostingRegressor
    # from sklearn.ensemble import AdaBoostRegressor
    # from sklearn.svm import SVC
    # from sklearn.utils import class_weight

    from xgboost import XGBRegressor
    # from sklearn.model_selection import StratifiedKFold
    # from sklearn.model_selection import cross_val_score
    # kfold = StratifiedKFold(n_splits=2, random_state=it)
    # results = cross_val_score(model, X, Y, cv=kfold)

    # model = XGBRegressor()
    # model.fit(x_train, y_train)
    ################################
    # model = gbm.train(params,
    #                   lgb_train,
    #                   num_boost_round=200,
    #                   valid_sets=lgb_eval,
    #                   early_stopping_rounds=20)
    #
    # y_pred = model.predict(x_test_2)


    ####

    # model = XGBRegressor(objective="binary:logistic", subsample=0.5,
    #              learning_rate=0.005, max_depth=8,
    #              min_child_weight=1.8, n_estimators=4000,
    #              reg_alpha=0.1, reg_lambda=0.3, gamma=0.01,
    #              silent=1, random_state=7, nthread=-1)

    model = XGBRegressor(objective="binary:logistic", eval_metric="auc", subsample=0.5,
                         learning_rate=0.005, max_depth=6,
                         min_child_weight=1.8, n_estimators=4000,
                         reg_alpha=0.1, reg_lambda=0.3, gamma=0.1,
                         silent=1, random_state=8, nthread=-1)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test_2)

    ####
    xg_predictions = [int(round(value)) for value in y_pred]
    print('Round validation ROCAUC, accuracy, recall, precision', roc_auc_score(y_test_2, y_pred),
          accuracy_score(y_test_2, xg_predictions), recall_score(y_test_2, xg_predictions),
          precision_score(y_test_2, xg_predictions))

    y_xg_1 = model.predict(test_X)
    test_set_results.append(y_xg_1)

test_set_results = np.array(test_set_results)
print('Results', test_set_results.shape)

##accuracy = accuracy_score(final_score, xg_predictions)
# print("Accuracy: %.2f%%" % (accuracy * 100.0))
print(test_set_results)
final_y = np.average(test_set_results, axis=0)
print(final_y.shape, final_y)

import pandas as pd

df = pd.DataFrame()
df["Predicted"] = final_y
df.to_csv('xg_save.csv', index_label="Id")
'''
# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
'''