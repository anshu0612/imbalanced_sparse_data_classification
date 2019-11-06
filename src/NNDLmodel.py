import os
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from keras import regularizers
from sklearn.utils import class_weight
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
# from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras.backend as K
from keras.preprocessing import sequence
from keras.models import Sequential, Model
from keras.layers import GRU, Concatenate, Dense, Dropout, Embedding, LSTM, Bidirectional, GlobalMaxPooling1D, Input, \
    BatchNormalization, Conv1D, Multiply, Activation, MaxPooling1D
from keras.regularizers import l2
from keras.optimizers import Adam, SGD
from sklearn.utils import shuffle

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
# plt.style.use('seaborn')

prefix = 'data'
labels = pd.read_csv(prefix + '/train_kaggle.csv')


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

sig_conv = Conv1D(128, (1), activation='sigmoid', padding='same', kernel_regularizer=regularizers.l2(0.0005))(X)
# rel_conv = Conv1D(64, (1), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.0005))(X)
# a = Multiply()([sig_conv, rel_conv])

# b_sig = Conv1D(filters=64, kernel_size=(2), strides=1, kernel_regularizer=regularizers.l2(0.0005), activation="sigmoid", padding="same")(X)
# b_relu = Conv1D(filters=64, kernel_size=(2), strides=1, kernel_regularizer=regularizers.l2(0.0005), activation="relu", padding="same")(X)
# b = Multiply()([b_sig, b_relu])

# X = Concatenate()([a, b])
# X = BatchNormalization()(X)
X = Bidirectional(LSTM(100))(sig_conv)
#X = LSTM(64)(sig_conv)

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


predictions = []
max_len = 40
batch_size = 512
no_epochs = 30
ml = 340

X_test = []
for fileno in range(10000):
    zero_mat = np.zeros((ml, 40))
    features = np.load(prefix + '/test/test/' + str(fileno) + '.npy')
    zero_mat[:features.shape[0], :] = features[:min(ml, features.shape[0]), :]
    X_test.append(zero_mat)
X_test = np.nan_to_num(np.array(X_test))

import math

# labels_dict : {ind_label: count_label}
# mu : parameter to tune

def create_class_weight(labels_dict, mu=0.15):
    total = np.sum(labels_dict.values())
    keys = labels_dict.keys()
    class_weight = dict()

    for key in keys:
        score = math.log(mu*total/float(labels_dict[key]))
        class_weight[key] = score if score > 1.0 else 1.0

    return class_weight

for i in range(10):
    X = []
    y = []
    ones = len(labels.loc[labels['label'] == 1])
    shuffled_labels = shuffle(labels)
    shuffled_y = np.array(shuffled_labels['label'])
    X_data = []
    for index, train_label in shuffled_labels.iterrows():
        label = train_label['label']
        zero_mat = np.zeros((ml, 40))

        if label == 0 and ones > 0:
            ones = ones - 0.85
        if ones <= 0 and label == 0:
            continue
        ## features is a (N, 40) matrix
        features = np.load(prefix + '/train/train/' + str(train_label['Id']) + '.npy')

        zero_mat[:features.shape[0], :] = features[:min(ml, features.shape[0]), :]

        X_data.append(zero_mat)
        y.append(label)

    X_data = np.nan_to_num(np.array(X_data))
    y = np.array(y)
    print(("=====>", X_data.shape))
    print("y shape", y.shape)

    ## Split into train and test datasets
    x_train, x_test, y_train, y_test = train_test_split(X_data, y, shuffle=True, test_size=0.20)
    print("~~~~~~", x_train)
    # a = np.arange(40)
    # np.random.shuffle(a)
    # rm = a[:5]
    # xr = np.delete(xr, rm, axis=2)
    # X_test_dup = np.delete(X_test_dup, rm, axis=2)

    model.compile(optimizer=Adam(lr=0.001, decay=1e-8), loss='binary_crossentropy',
                  metrics=['accuracy', f1_m, precision_m, recall_m])
    generator2 = generate_data(x_train, y_train, batch_size)

    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=10, verbose=1, mode='min')
    terminate_on_nan = TerminateOnNaN()
    model_checkpoint = ModelCheckpoint("cp1", monitor='loss', save_best_only=True, mode='min')
    early_stopping = EarlyStopping(monitor='recall_m', patience=10, mode='auto')

    class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
    model.fit_generator(
        generator2,
        steps_per_epoch=math.ceil(len(x_train) / batch_size),
        epochs=no_epochs,
        shuffle=True,
        #class_weight=class_weights,
        verbose=1,
        # initial_epoch=86,
        validation_data=(x_test, y_test),
        callbacks=([model_checkpoint, terminate_on_nan, reduce_lr, early_stopping]))
    print(model.evaluate(x_test, y_test, verbose=0))
    loss, accuracy, f1_score, precision, recall = model.evaluate(x_test, y_test, verbose=0)
    pred = model.predict(X_test)
    predictions.append(pred)

predictions = np.array(predictions)
print("Ensemble labels shape:", predictions.shape)
predictions = np.mean(predictions, axis=0)

pred = pd.DataFrame(data=predictions, index=[i for i in range(predictions.shape[0])], columns=["Predicted"])
pred.index.name = 'Id'
pred.to_csv('rand_f_re.csv', index=True)
