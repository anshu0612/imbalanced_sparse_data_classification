import os
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from keras import regularizers
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
#from sklearn.feature_selection import VarianceThreshold
from keras.models import load_model
#from sklearn.svm import SVC
#from sklearn.tree import DecisionTreeRegressor
#from sklearn.ensemble import RandomForestRegressor
#from sklearn.ensemble import GradientBoostingRegressor

from sklearn.utils import class_weight
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
#from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

from xgboost import XGBRegressor
# import lightgbm

import tensorflow as tf
import keras.backend as K
from keras.preprocessing import sequence
from keras.models import Sequential, Model
from keras.layers import Concatenate, Dense, Dropout, Embedding, LSTM, Bidirectional, GlobalMaxPooling1D, Input, BatchNormalization, Conv1D, Multiply, Activation, MaxPooling1D
from keras.regularizers import l2
from keras.optimizers import Adam, SGD

from keras.callbacks.callbacks import TerminateOnNaN, ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau, \
    EarlyStopping

def classification_evaluation(y_ture, y_pred):
  acc = accuracy_score(y_ture, (y_pred>0.5).astype('int'))
  auc = roc_auc_score(y_ture, y_pred)
  fpr, tpr, thresholds = roc_curve(y_ture, y_pred)

  print('Accuracy:', acc)
  print('ROC AUC Score:', auc)

  plt.plot([0, 1], [0, 1], linestyle='--')
  plt.plot(fpr, tpr, marker='.')
  plt.xlabel('FPR')
  plt.ylabel('Recall rate')
  plt.show()


plt.style.use('seaborn')

# max_len = 340
#336
batch_size = 64
#128
train_samples = 30336
# 30336
test_samples = 10000
#10000
# no_epochs = 88

labels = pd.read_csv("data/train_kaggle.csv")
ones = len(labels.loc[labels['label']==1])
zeros = ones
X_t = []
y_t = []
print(ones, zeros)

for index, train_label in labels.iterrows():
    label = train_label['label']
    zero_mat = np.zeros((50, 40))
    data = np.load("data/train/train/" + str(train_label['Id']) + '.npy')
    zero_mat[:min(50, data.shape[0]), :] = data[:min(50, data.shape[0]), :]
    X_t.append(zero_mat)
    y_t.append(label)

x_sampled = []
y_sampled = []
X_t = np.array(X_t)
y_t = np.array(y_t)

from sklearn.utils import shuffle
X_t, y_t = shuffle(X_t, y_t, random_state=0)
print("Stage 1", X_t.shape, y_t.shape)
# shuffle data
for index in range(y_t.shape[0]):
    label = y_t[index]
    if label == 0 and zeros > 0:
        zeros = zeros - 1
    if label == 1 and ones > 0:
        ones = ones - 1
    if (zeros == 0 and label == 0) or (ones == 0 and label == 1):
        continue
    # df1 = pd.DataFrame(data=X_t[index])
    # for feature in range(40):
    #     mode = df1[feature].mode()
    #     df1[feature].fillna(mode, inplace=True)
    # m = np.array(df1)
    m = X_t[index].reshape((-1,))
    print("==>",m.shape)
    x_sampled.append(m)
    y_sampled.append(y_t[index])

X = np.nan_to_num(np.array(x_sampled))
y = np.array(y_sampled)

print("Stage 2", X.shape, y.shape)
X_train, X_val, Y_train, Y_val = train_test_split(X, y, shuffle=True, test_size=0.15)

pca = PCA(n_components=100)
X_train_new = pca.fit_transform(X_train)
X_val_new = pca.transform(X_val)
print(X_train_new.shape)

xgb_model = XGBRegressor(objective="binary:logistic", max_depth=8, min_samples_leaf=2, n_estimators=300, random_state=42)
xgb_model.fit(X_train_new, Y_train)
y_pred = xgb_model.predict(X_val_new)
classification_evaluation(Y_val, y_pred)

#
# def focal_loss(y_true, y_pred):
#     gamma = 2.0
#     alpha = 0.25
#     pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
#     pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
#     return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))
#
# #model = load_model("cp1")
# def recall_m(y_true, y_pred):
#
#     true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#     possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
#     recall = true_positives / (possible_positives + K.epsilon())
#     return recall
#
# def precision_m(y_true, y_pred):
#     true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#     predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
#     precision = true_positives / (predicted_positives + K.epsilon())
#     return precision
#
# def f1_m(y_true, y_pred):
#     precision = precision_m(y_true, y_pred)
#     recall = recall_m(y_true, y_pred)
#     return 2*((precision*recall)/(precision+recall+K.epsilon()))


# # print("EVALUATION loss:", loss,"accuracy:",  accuracy, "f1_score:", f1_score, "precision:",precision, "recall:",recall)
# '''
#     STEP 4 : Prepare test samples
# '''
# X_test = []
# for i in range(0, test_samples):
#     data = np.load("data/test/test/" + str(i) + ".npy")
#     zero_mat = np.zeros((50, 40))
#     zero_mat[:data.shape[0], :] = data[:min(50, data.shape[0]), :]
#     df1 = pd.DataFrame(data=zero_mat)
#     for feature in range(40):
#         mod = df1[feature].mode()
#         df1[feature].fillna(mod, inplace=True)
#
#     zero_mat = np.array(df1)
#     zero_mat = np.delete(zero_mat, [2, 34, 16, 10], axis=1)
#
#
#     X_test.append(np.array(zero_mat))
#
# X_test = np.nan_to_num(np.array(X_test))
# print(X_test.shape)
# #model = load_model("cp1")
# #model = load_model("cp1")
#
# pred = model.predict(X_test)
# print(pred.shape, pred)
# pred = pd.DataFrame(data=pred, index=[i for i in range(pred.shape[0])], columns=["Predicted"])
# pred.index.name = 'Id'
# pred.to_csv('rnn_v11.csv', index=True)
