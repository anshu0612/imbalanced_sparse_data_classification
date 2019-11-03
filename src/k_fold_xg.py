import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBRegressor

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

kfolds = StratifiedKFold(n_splits=4,  shuffle=True, random_state=1)

for it in range(5):
    print('Starting XGBoost Iteration ', it)
    X = []
    y = []
    ones = len(labels.loc[labels['label'] == 1])
    shuffled_labels = shuffle(labels)
    shuffled_y = np.array(shuffled_labels['label'])
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
        features = np.load(prefix_path + '/train/train/' + str(train_label['Id']) + '.npy')
        sparse_x, dense_x = __preprocess_feature(features)
        feature_avg = np.average(features, axis=0)
        sparse_means = np.nanmean(np.where(sparse_x != 0, sparse_x, np.nan), axis=0)
        # u, l = np.percentile(sparse_means, [1, 99])
        # y = np.clip(sparse_means, u, l)
        X_sparse.append(sparse_means)
        y.append(label)
    X_tn = np.nan_to_num(np.array(X_sparse))
    y_tn = np.array(y)
    # x_train, x_test, y_train, y_test = train_test_split(X_sparse, y, test_size=0.20)
    f = 0
    for train, test in kfolds.split(X_tn, y_tn):
        print('Starting XGB training for fold:', f + 1)
        model = XGBRegressor(objective="binary:logistic", eval_metric="auc", subsample=0.5,
                             learning_rate=0.005, max_depth=8,
                             min_child_weight=5, n_estimators=3000,
                             reg_alpha=0.1, reg_lambda=0.3, gamma=0.1,
                             silent=1, random_state=8, nthread=-1)
        model.fit(X_tn[train], y_tn[train])
        y_pred = model.predict(X_tn[test])
        xg_predictions = [int(round(value)) for value in y_pred]
        print('Round validation ROCAUC, accuracy, recall, precision', roc_auc_score(y_tn[test], y_pred),
              accuracy_score(y_tn[test], xg_predictions), recall_score(y_tn[test], xg_predictions),
              precision_score(y_tn[test], xg_predictions))

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