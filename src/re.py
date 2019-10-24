import os
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from keras import regularizers
from sklearn.model_selection import train_test_split
# from sklearn.decomposition import PCA
# from sklearn.feature_selection import VarianceThreshold
from keras.models import load_model
# from sklearn.svm import SVC
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.ensemble import GradientBoostingRegressor
from sklearn.utils import class_weight
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
# from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

# from xgboost import XGBRegressor
# import lightgbm

import tensorflow as tf
import keras.backend as K
from keras.preprocessing import sequence
from keras.models import Sequential, Model
from keras.layers import GRU, Concatenate, Dense, Dropout, Embedding, LSTM, Bidirectional, GlobalMaxPooling1D, Input, \
    BatchNormalization, Conv1D, Multiply, Activation, MaxPooling1D
from keras.regularizers import l2
from keras.optimizers import Adam, SGD

from keras.callbacks.callbacks import TerminateOnNaN, ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau, \
    EarlyStopping


def classification_evaluation(y_ture, y_pred):
    acc = accuracy_score(y_ture, (y_pred > 0.5).astype('int'))
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
# 336
batch_size = 64
# 128
train_samples = 30336
# 30336
test_samples = 100
# 10000
no_epochs = 88

labels = pd.read_csv("data/train_kaggle.csv")
X_t = []
y_t = []
ml = 50
for index, train_label in labels.iterrows():
    label = train_label['label']
    zero_mat = np.zeros((ml, 40))
    data = np.load("data/train/train/" + str(train_label['Id']) + '.npy')
    # print("STEP 1:", data.shape)
    # zero_mat[:data.shape[0], :] = data[:min(50, data.shape[0]), :]

    df1 = pd.DataFrame(data=data)
    Q1 = df1.quantile(0.25)
    Q3 = df1.quantile(0.75)
    IQR = Q3 - Q1
    df1 = df1[~((df1 < (Q1 - 1.5 * IQR)) | (df1 > (Q3 + 1.5 * IQR))).any(axis=1)]
    data = np.array(df1)
    # print(data.shape[0])
    # if(data.shape[0] < mi):
    #    mi = data.shape[0]
    # print("STEP 2:", data.shape)
    zero_mat[:data.shape[0], :] = data[:min(ml, data.shape[0]), :]
    # zero_mat[:data.shape[0], :] = data
    # print("STEP 2:", data.shape)

    X_t.append(zero_mat)
    y_t.append(label)
# print("*****************", mi)
X_t = np.nan_to_num(np.array(X_t))
y_t = np.array(y_t)

from sklearn.utils import shuffle
print("Stage 1", X_t.shape, y_t.shape)


# rm = []
mm = len(labels.loc[labels['label'] == 1])
rm = []
def sample():
    ones = mm
    zeros = mm
    XX, yy = shuffle(X_t, y_t, random_state=0)
    x_sampled = []
    y_sampled = []
    # a = np.arange(40)
    # np.random.shuffle(a)
    # rm = a[:5]
    for index in range(yy.shape[0]):
        label = yy[index]
        if label == 0:
            if zeros > 0:
                zeros = zeros - 1
            else:
                continue
        if label == 1:
            if ones > 0:
                ones = ones - 1
            else:
                continue
        if zeros < 0 and ones < 0:
            break
        # for feature in range(40):
        # average_value = np.nanmean(X_t[index][:, feature])
        # print(average_value)
        # X_t[index][:, feature] = np.nan_to_num(X_t[index][:, feature], nan=average_value)
        # X_t[index][:, feature] = np.nan_to_num(X_t[index][:, feature], nan=0)
        m = np.delete(XX[index], rm, axis=1)
        x_sampled.append(m)
        y_sampled.append(yy[index])

    return np.array(x_sampled), np.array(y_sampled)
#
# X = np.nan_to_num(np.array(x_sampled))
# y = np.array(y_sampled)

# print("Stage 2", X.shape, y.shape)
# print(y)
# df = pd.read_csv("data/train_kaggle.csv")
# Y_t = df[:train_samples]
# y = Y_t.values


# print("Trainig set", X_train, X_val)

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


data_input = Input(shape=(None, 40))
X = BatchNormalization()(data_input)

sig_conv = Conv1D(64, (1), activation='sigmoid', padding='same', kernel_regularizer=regularizers.l2(0.0005))(X)
# rel_conv = Conv1D(64, (1), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.0005))(X)
# a = Multiply()([sig_conv, rel_conv])

# b_sig = Conv1D(filters=64, kernel_size=(2), strides=1, kernel_regularizer=regularizers.l2(0.0005), activation="sigmoid", padding="same")(X)
# b_relu = Conv1D(filters=64, kernel_size=(2), strides=1, kernel_regularizer=regularizers.l2(0.0005), activation="relu", padding="same")(X)
# b = Multiply()([b_sig, b_relu])

# X = Concatenate()([a, b])
# X = BatchNormalization()(X)
X = Bidirectional(LSTM(64))(sig_conv)
# X = Bidirectional(LSTM(64))(X)
# X = GlobalMaxPooling1D()(X)
X = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.0005))(X)
X = Dropout(0.5)(X)
X = Dense(1, kernel_regularizer=regularizers.l2(0.0005))(X)
X = Activation("sigmoid")(X)
model = Model(input=data_input, output=X)


def focal_loss(y_true, y_pred):
    gamma = 2.0
    alpha = 0.25
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.sum(
        (1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))


# model = load_model("cp1")
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
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


# def roc_auc(y_true, y_pred):
#     print(type(y_true), type(y_pred))
#     return roc_auc_score(y_true, y_pred)

def f1_loss(y_true, y_pred):
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1 - y_true) * (1 - y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)
    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())
    f1 = 2 * p * r / (p + r + K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return 1 - K.mean(f1)


X_test = []
for i in range(0, test_samples):
    data = np.load("data/test/test/" + str(i) + ".npy")
    zero_mat = np.zeros((ml, 40))

    df1 = pd.DataFrame(data=data)
    Q1 = df1.quantile(0.25)
    Q3 = df1.quantile(0.75)
    IQR = Q3 - Q1
    df1 = df1[~((df1 < (Q1 - 1.5 * IQR)) | (df1 > (Q3 + 1.5 * IQR))).any(axis=1)]
    data = np.array(df1)

    # for feature in range(40):
    # average_value = np.nanmean(zero_mat[:, feature])
    # zero_mat[:, feature]= np.nan_to_num(zero_mat[:, feature], nan=average_value)
    # zero_mat[:, feature] = np.nan_to_num(zero_mat[:, feature], nan=0)
    zero_mat[:data.shape[0], :] = data[:min(ml, data.shape[0]), :]
    zero_mat = np.delete(zero_mat, rm, axis=1)
    # 11, 33, 35
    # 1,3, 4, 15, 17,  22, 24, 36
    # 1, 3, 4, 6, 8, 15, 17, 18, 20, 22, 23, 24, 29, 31, 36
    X_test.append(zero_mat)

X_test = np.nan_to_num(np.array(X_test))

predictions = []
for i in range(2):
    xr, yr =  sample()
    X_train, X_val, Y_train, Y_val = train_test_split(xr, yr, shuffle=True, test_size=0.15)

    print("Trainig set shapes", X_train.shape, Y_train, X_val.shape, Y_val.shape)
    model.compile(optimizer=Adam(lr=0.001, decay=1e-8), loss="binary_crossentropy",
                  metrics=['accuracy', f1_m, precision_m, recall_m])
    generator2 = generate_data(X_train, Y_train, batch_size)

    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=10, verbose=1, mode='min')
    terminate_on_nan = TerminateOnNaN()
    model_checkpoint = ModelCheckpoint("cp1", monitor='loss', save_best_only=True, mode='min')
    early_stopping = EarlyStopping(monitor='recall_m', patience=10, mode='auto')

    class_weights = class_weight.compute_class_weight('balanced', np.unique(Y_train), Y_train)

    model.fit_generator(
        generator2,
        steps_per_epoch=math.ceil(len(X_train) / batch_size),
        epochs=no_epochs,
        shuffle=True,
        class_weight=class_weights,
        verbose=1,
        # initial_epoch=86,
        validation_data=(X_val, Y_val),
        callbacks=([model_checkpoint, terminate_on_nan, reduce_lr, early_stopping]))

    loss, accuracy, f1_score, precision, recall = model.evaluate(X_val, Y_val, verbose=0)
    print("EVALUATION loss:", loss, "accuracy:", accuracy, "f1_score:", f1_score, "precision:", precision, "recall:",
          recall)

    pred = model.predict(X_test)
    print(pred.shape, pred)
    predictions.append(pred)

predictions = np.array(predictions)
print("Ensemble labels shape:", predictions.shape)
predictions = np.mean(predictions, axis=0)

pred = pd.DataFrame(data=predictions, index=[i for i in range(predictions.shape[0])], columns=["Predicted"])
pred.index.name = 'Id'
pred.to_csv('re.csv', index=True)
