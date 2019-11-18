import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor, CatBoostClassifier
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
prefix_path = 'data'

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

for it in range(6):
    print('Starting XGBoost Iteration ', it)
    X = []
    y = []
    ## ones count kept to balance number of zeros and ones in data to be equal
    ones = len(labels.loc[labels['label'] == 1])

    max_len = 340
    batch_size = 512
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
        assert feature_avg.shape == (features.shape[1],)
        ##if len(X) == 0:
        #    X = np.array([feature_avg])
        # else:
        #    X = np.concatenate((X, [feature_avg]))

        sparse_means = np.nanmean(np.where(sparse_x != 0, sparse_x, np.nan), axis=0)
        X_sparse.append(sparse_means)
        y.append(label)

    X_sparse = np.nan_to_num(np.array(X_sparse))
    y = np.array(y)
    # from imblearn.under_sampling import RandomUnderSampler
    # cc = RandomUnderSampler(random_state=0)
    # print("UNDERSAMPLING IN PROGRESS===== **** ")
    # X_sparse, y = cc.fit_resample(X_sparse, y)
    X_sparse = np.delete(X_sparse, [4, 10, 16, 25, 28, 30, 34], axis=1)

    ## Split into train and test datasets
    x_train, x_test, y_train, y_test = train_test_split(X_sparse, y, test_size=0.20)

    # from imblearn.over_sampling import SMOTE
    # sm = SMOTE(random_state=27, ratio=1.0)
    # print("Before===>", x_train.shape)
    # x_train, y_train = sm.fit_sample(x_train, y_train)
    # print("After===>", x_train.shape)
    # x_test_2 = np.concatenate([x_test, zero_test])
    # y_test_2 = np.concatenate([y_test, zero_test_y])
    x_test_2 = np.concatenate([x_test])
    y_test_2 = np.concatenate([y_test])

    # weight_ratio = float(len(y_train[y_train == 0]))/float(len(y_train[y_train ==
    # 1]))
    # w_array = np.array([1.0]*y_train.shape[0])
    # w_array[y_train==1] = 8
    # w_array[y_train==0] = 1

    # print('Weights', w_array.shape, w_array)
    # class_weight = {0: 1., 1: 10}

    print('X_train Shape', x_train.shape)
    print('X_test shape', x_test.shape)
    print('y_train Shape', y_train.shape)
    print('y_test shape', y_test.shape)


    model = CatBoostClassifier(iterations=800, learning_rate=0.01, l2_leaf_reg=3.5, depth=8, rsm=0.98,
                               loss_function='Logloss', eval_metric='AUC', use_best_model=True, random_seed=42)

    model.fit(x_train, y_train, cat_features=[], eval_set=(x_test_2, y_test_2))

    y_pred = model.predict_proba(x_test_2)
    print(y_pred.shape)
    y_pred = np.array(y_pred[:, 1])
    xg_predictions = [int(np.round(value)) for value in y_pred]
    print('Round validation ROCAUC, accuracy, recall, precision', roc_auc_score(y_test_2, y_pred),
          accuracy_score(y_test_2, xg_predictions), recall_score(y_test_2, xg_predictions),
          precision_score(y_test_2, xg_predictions))

    y_xg_1 = model.predict_proba(test_X)
    test_set_results.append(y_xg_1[:, 1])

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
df.to_csv('cat_final.csv', index_label="Id")
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