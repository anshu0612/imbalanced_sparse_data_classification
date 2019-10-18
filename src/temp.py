import os
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from keras import regularizers
from sklearn.model_selection import train_test_split
#from sklearn.decomposition import PCA
#from sklearn.feature_selection import VarianceThreshold

#from sklearn.svm import SVC
#from sklearn.tree import DecisionTreeRegressor
#from sklearn.ensemble import RandomForestRegressor
#from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
#from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

#from xgboost import XGBRegressor
# import lightgbm

import tensorflow as tf
import keras.backend as K
from keras.preprocessing import sequence
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional, GlobalMaxPooling1D, Input, BatchNormalization, Conv1D, Multiply, Activation, MaxPooling1D
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

max_len = 340
#336
batch_size = 64
#128
train_samples = 30336
# 30336
test_samples = 10000
#10000
no_epochs = 100

labels = pd.read_csv("data/train_kaggle.csv")
ones = len(labels.loc[labels['label']==1])
zeros = ones
X_t = []
y_t = []
print(ones, zeros)
for index, train_label in labels.iterrows():
    label = train_label['label']
    if label == 0 and zeros > 0:
        zeros = zeros - 1
    if label == 1 and ones > 0:
        ones = ones - 1
    if (zeros == 0 and label == 0) or (ones == 0 and label == 1):
        continue
    zero_mat = np.zeros((max_len, 40))
    data = np.load("data/train/train/" + str(train_label['Id']) + '.npy')
    zero_mat[:data.shape[0], :] = data
    for feature in range(40):
        average_value = np.nanmean(zero_mat[:, feature])
        zero_mat[:, feature]= np.nan_to_num(zero_mat[:, feature], nan=average_value)
    zero_mat = np.delete(zero_mat, [1, 3, 4, 6, 8, 18, 23, 29, 31], axis=1)
    X_t.append(zero_mat)
    y_t.append(label)

X = np.array(X_t)
y = np.array(y_t)
# df = pd.read_csv("data/train_kaggle.csv")
# Y_t = df[:train_samples]
# y = Y_t.values
print(X.shape, y.shape)
# print("FINALLY---->", X[0], y)

X_train, X_val, Y_train, Y_val = train_test_split(X, y, shuffle=True, test_size=0.20)

# print("Trainig set", X_train, X_val)
print("Trainig set shapes", X_train.shape, Y_train,  X_val.shape, Y_val.shape)


def generate_data(x_data, y_data, b_size):
    samples_per_epoch = x_data.shape[0]
    number_of_batches = samples_per_epoch / b_size
    counter = 0
    while 1:
        x_batch = np.array(x_data[batch_size * counter:batch_size * (counter + 1)])
        y_batch = np.array(y_data[batch_size * counter:batch_size * (counter + 1)])
        counter += 1
        yield x_batch, y_batch

        if counter >= number_of_batches:
            counter = 0


data_input = Input(shape=(None, 31))

X = BatchNormalization()(data_input)

sig_conv = Conv1D(40, (1), activation='sigmoid', padding='same')(X)
rel_conv = Conv1D(40, (1), activation='relu', padding='same')(X)
X = Multiply()([sig_conv, rel_conv])
X = Bidirectional(LSTM(64))(X)
#X = GlobalMaxPooling1D()(X)
X = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.0005))(X)
X = Dropout(0.5)(X)
X = Dense(1, kernel_regularizer=regularizers.l2(0.0005))(X)
X = Activation("sigmoid")(X)
#dense_1 = Dense(16, activation='relu')(lstm)
#dense_2 = Dense(1)(dense_1)
#out = Activation('sigmoid')(dense_2)
model = Model(input=data_input, output=X)

def focal_loss(y_true, y_pred):
    gamma = 2.0
    alpha = 0.25
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

model.compile(optimizer=SGD(lr=0.001), loss=[focal_loss], metrics=['accuracy', f1_m,precision_m, recall_m])

generator2 = generate_data(X_train, Y_train, batch_size)

reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=8, verbose=1, mode='min')
terminate_on_nan = TerminateOnNaN()
model_checkpoint = ModelCheckpoint("cp1", monitor='loss', save_best_only=True, mode='min')
early_stopping = EarlyStopping(monitor=['loss'], patience=4, mode='auto')


model.fit_generator(
    generator2,
    steps_per_epoch=math.ceil(len(X_train) / batch_size),
    epochs=no_epochs,
    shuffle=True,
    verbose=1,
    #initial_epoch=36,
    validation_data=(X_val, Y_val),
    callbacks=([model_checkpoint]))

loss, accuracy, f1_score, precision, recall = model.evaluate(X_val, Y_val, verbose=0)
print("EVALUATION loss:", loss,"accuracy:",  accuracy, "f1_score:", f1_score, "precision:",precision, "recall:",recall)
'''
    STEP 4 : Prepare test samples
'''
X_test = []
for i in range(0, test_samples):
    data = np.load("data/test/test/" + str(i) + ".npy")
    zero_mat = np.zeros((max_len, 40))
    zero_mat[:data.shape[0], :] = data
    for feature in range(40):
        average_value = np.nanmean(zero_mat[:, feature])
        zero_mat[:, feature]= np.nan_to_num(zero_mat[:, feature], nan=average_value)
    zero_mat = np.delete(zero_mat, [1, 3, 4, 6, 8, 18, 23, 29, 31], axis=1)
    X_test.append(zero_mat)

X_test = np.array(X_test)
print(X_test.shape)

pred = model.predict(X_test)
print(pred.shape, pred)
pred = pd.DataFrame(data=pred, index=[i for i in range(pred.shape[0])], columns=["Predicted"])
pred.index.name = 'Id'
pred.to_csv('rnn_v3.csv', index=True)
