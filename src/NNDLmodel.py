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
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras.backend as K
from keras.preprocessing import sequence
from keras.models import Sequential, Model
from keras.layers import TimeDistributed, GRU, Concatenate, Dense, Dropout, Embedding, LSTM, Bidirectional, GlobalMaxPooling1D, Input, \
    BatchNormalization, Conv1D, Multiply, Activation, MaxPooling1D
from keras.layers import Dense, Activation, Dropout, Input, Multiply, LSTM, Conv1D, Flatten, \
            RepeatVector, Permute, Lambda
from keras.regularizers import l2
from keras.optimizers import Adam, SGD
from sklearn.utils import shuffle
from scipy import stats
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

sparse_index = [i for i in range(40)]
# sparse_index = [i for i in sparse_index if i not in [4, 10, 25]]
dense_index = [2, 3, 5, 7, 11, 12, 13, 15, 17, 18, 20, 24, 29, 33, 35, 37, 39]


# sparse_index = [2, 3, 5, 7, 11, 12, 13, 15, 17, 18, 20, 24, 29, 33, 35, 37, 39]

def __preprocess_feature(feat):
    sparse_x = feat[:, sparse_index]
    dense_x = feat[:, dense_index]
    return sparse_x


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

predictions = []
max_len = 40
batch_size = 128
no_epochs = 100
ml = 150


def mymodel():
    #data_input = Input(shape=(None, 40))
    #X = BatchNormalization()(data_input)

    X_input = Input(shape=(ml, 40))

    X = BatchNormalization()(X_input)
    a = Conv1D(filters=64, kernel_size=(1), strides=1, kernel_regularizer=regularizers.l2(0.0005), activation="sigmoid", padding="same")(X)
    b = Conv1D(filters=64, kernel_size=(2), strides=1, kernel_regularizer=regularizers.l2(0.0005),activation="sigmoid", padding="same")(X)
    X = Concatenate()([a, b])                                                            # Normalization 2
    X = BatchNormalization()(a)

                                               # BidirectionalLSTM
    X = Bidirectional(LSTM(32, return_sequences=True))(X)
    X = Bidirectional(LSTM(32, return_sequences=True))(X)
    attention = Dense(1, activation='tanh')(X)
    attention = Flatten()(attention)
    attention = Activation('softmax')(attention)
    attention = RepeatVector(64)(attention)
    attention = Permute([2, 1])(attention)
    s_r = Multiply()([X, attention])
    s_r = Lambda(lambda xin: K.sum(xin, axis=-2), output_shape=(64,))(s_r)

    #X = Bidirectional(LSTM(32))(s_r)
    #X = GlobalMaxPooling1D()(X)
    #X = Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.0005))(s_r)
    #X = Dense(32, activation='tanh', kernel_regularizer=regularizers.l2(0.0005))(X)
    X = BatchNormalization()(s_r)
    X = Dropout(0.2)(X)
    
    X = Dense(1, kernel_regularizer=regularizers.l2(0.0005))(X)
    X = Activation("sigmoid")(X)
    model = Model(input=X_input, output=X)
    return model

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


#predictions = []
#max_len = 40
#batch_size = 128
#no_epochs = 100
#ml = 40

X_test = []
for fileno in range(10000):
    zero_mat = np.zeros((ml, 40))
    features = np.load(prefix + '/test/test/' + str(fileno) + '.npy')
    
    #sparse_x = __preprocess_feature(features)
                    #sparse_means = np.nanmean(np.where(sparse_x != 0, sparse_x, np.nan), axis=0)
    #sparse_max = np.max(np.where(sparse_x != 0, sparse_x, np.nan), axis=0)
    #sparse_medians = np.nanmedian(np.where(sparse_x != 0, sparse_x, np.nan), axis=0)
                                            #sparse_nans = np.count_nonzero(np.isnan(sparse_x), axis=0)
    #sparse_mode = stats.mode(sparse_x)
    #sparse_mode = sparse_mode[0][0]
    #sp = np.stack([sparse_mode, sparse_max, sparse_medians, sparse_x[0], sparse_x[-1]], axis=0)
               # print("kya hai shape", sp.shape)
    #sp = sp.reshape(1, sp.shape[0])

   # sp = np.stack([sparse_mode, sparse_max, sparse_medians, sparse_x[0], sparse_x[-1]], axis=0)
    
    zero_mat[:features.shape[0], :] = features[:min(ml, features.shape[0]), :]
    X_test.append(zero_mat)
X_test = np.nan_to_num(np.array(X_test), 0)

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

for i in range(30):
    print("Iteration no.:", i+1)
    X = []
    y = []
    ones = len(labels.loc[labels['label'] == 1])
    shuffled_labels = shuffle(labels)
    shuffled_y = np.array(shuffled_labels['label'])
    X_data = []
    for index, train_label in shuffled_labels.iterrows():
        label = train_label['label']
        #zero_mat = np.zeros((ml, 40))

        if label == 0 and ones > 0:
            ones = ones - 0.85
        if ones <= 0 and label == 0:
            continue
        ## features is a (N, 40) matrix
        zero_mat = np.zeros((ml, 40))
        features = np.load(prefix + '/train/train/' + str(train_label['Id']) + '.npy')
        #sparse_x = __preprocess_feature(features)
         
        #features = np.log(1 + features)
        #sparse_means = np.nanmean(np.where(sparse_x != 0, sparse_x, np.nan), axis=0)
        #sparse_max = np.max(np.where(sparse_x != 0, sparse_x, np.nan), axis=0)
        #sparse_min = np.min(np.where(sparse_x != 0, sparse_x, np.nan), axis=0)
        #sparse_medians = np.nanmedian(np.where(sparse_x != 0, sparse_x, np.nan), axis=0)
        #sparse_nans = np.count_nonzero(np.isnan(sparse_x), axis=0)
        #sparse_mode = stats.mode(sparse_x)
        #sparse_mode = sparse_mode[0][0]
        
        #sp = np.stack([sparse_mode], axis=0)

        #sp = np.stack([ sparse_x[0], sparse_mode, sparse_max, sparse_medians, sparse_x[-1]], axis=0)
        #print("shape before::::", sp.shape)
        #from sklearn.impute import SimpleImputer
        #df2 = pd.DataFrame(data=sp)
        #imp_mean = SimpleImputer( strategy='mean') #for median imputation replace 'mean' with 'median'
        #imp_mean.fit(df2)
        #imputed_train_df = imp_mean.transform(df2)
        #sp = np.array(imputed_train_df)
       # print("kya hai shape", sp.shape)
        #sp = sp.reshape(1, sp.shape[0])
        #print("shape::::", sp.shape)
        #df1 = pd.DataFrame(data=sp)
        #Q1 = df1.quantile(0.25)
        #Q3 = df1.quantile(0.75)
        #IQR = Q3 - Q1
        #df1 = df1[~((df1 < (Q1 - 1.5 * IQR)) | (df1 > (Q3 + 1.5 * IQR))).any(axis=1)]
        #features = np.array(df1)
        
        zero_mat[:features.shape[0], :] = features[:min(ml, features.shape[0]), :]
        X_data.append(zero_mat)
        y.append(label)
    
    X_data = np.nan_to_num(np.array(X_data), 0)
    #print(X_data)
    #X_data = np.log(1 + X_data)
    y = np.array(y)
    print(("=====>", X_data.shape))
    print("y shape", y.shape)

    ## Split into train and test datasets
    x_train, x_test, y_train, y_test = train_test_split(X_data, y, shuffle=True, test_size=0.20)
    #print("~~~~~~", x_train)
    # a = np.arange(40)
    # np.random.shuffle(a)
    # rm = a[:5]
    # xr = np.delete(xr, rm, axis=2)
    # X_test_dup = np.delete(X_test_dup, rm, axis=2)
    model = mymodel()
    model.compile(optimizer=Adam(lr=0.001, decay=1e-8), loss="binary_crossentropy",
                  metrics=['accuracy', f1_m, precision_m, recall_m])
    generator2 = generate_data(x_train, y_train, batch_size)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, mode='min')
    terminate_on_nan = TerminateOnNaN()
    #model_checkpoint = ModelCheckpoint("cp1", monitor='loss', save_best_only=True, mode='min')
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, mode='auto')

    class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
    model.fit_generator(
        generator2,
        steps_per_epoch=math.ceil(len(x_train) / batch_size),
        epochs=no_epochs,
        shuffle=True,
        class_weight=class_weights,
        verbose=1,
        #initial_epoch=86,
        validation_data=(x_test, y_test),
        callbacks=([terminate_on_nan, reduce_lr, early_stopping]))
    #model.fit(x=x_train, y=y_train, epochs=100, batch_size=128, shuffle=True, class_weight=class_weights, callbacks=([terminate_on_nan, reduce_lr, early_stopping]))
    
    #print(model.evaluate(x_test, y_test, verbose=0))
    #loss, accuracy, f1_score, precision, recall = model.evaluate(x_test, y_test, verbose=0)
    y_pred = model.predict(x_test)
    xg_predictions = [int(round(value)) for value in y_pred]
    print('Round validation ROCAUC, accuracy, recall, precision', roc_auc_score(y_test, y_pred), accuracy_score(y_test, xg_predictions), recall_score(y_test, xg_predictions),
            precision_score(y_test, xg_predictions))
    
    pred = model.predict(X_test)
    predictions.append(pred)

predictions = np.array(predictions)
print("Ensemble labels shape:", predictions.shape)
predictions = np.mean(predictions, axis=0)
print("~~~~~", predictions)
pred = pd.DataFrame(data=predictions, index=[i for i in range(predictions.shape[0])], columns=["Predicted"])
pred.index.name = 'Id'
pred.to_csv('rnn_v11_100_itr.csv', index=True)
