import numpy as np
import math
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from keras import regularizers
from keras.models import Model, load_model
from keras.layers import Dense, Activation, Dropout, Input, Multiply, TimeDistributed, LSTM, Conv1D, Flatten, \
    RepeatVector, Permute, Lambda
from keras.layers import Bidirectional, BatchNormalization, Concatenate, GlobalMaxPooling1D
from keras.optimizers import Adam, SGD
from keras.losses import binary_crossentropy
from keras.callbacks.callbacks import TerminateOnNaN, ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau, \
    EarlyStopping
from keras import backend as K
import tensorflow as tf

'''
    STEP 1 : Getting training sample
'''
train_samples = 18622
# 18622
test_samples = 6051
# 6051
batch_size = 128
no_epochs = 37
n_x = 102

# max length of a sequence
max_Tx = 1000
X_t = []
# 0.98874 0.98874 0.98874
for i in range(0, train_samples):
    data = np.load("data/train/train/" + str(i) + ".npy")
    # zero_mat = np.zeros((max_Tx, n_x))
    # zero_mat.put(i, data)
    zero_mat = np.zeros((max_Tx, 102))
    zero_mat[:data.shape[0], :] = data
    X_t.append(zero_mat)

X_t = np.array(X_t)
# df = X_train.values
# y_4 =  df.loc[df['Id'] == 200, ['Label']]
df = pd.read_csv("data/train_kaggle.csv", usecols=["Label"])
Y_t = df[:train_samples]
Y_t = Y_t.values
# usecols=["Label"]

# kfolds = StratifiedKFold(n_splits=4, shuffle=True, random_state=1)
# list(StratifiedKFold(n_splits=4, shuffle=False, random_state=1).split(X_train, Y_train))
print(X_t.shape, Y_t.shape)

X_train, X_val, Y_train, Y_val = train_test_split(X_t, Y_t, test_size=0.20)

print("Alright new thing###:", X_train.shape, X_val.shape, Y_train.shape, Y_val.shape)
'''
    STEP 2 : model
'''


def malware_detection_model(input_shape):
    X_input = Input(shape=input_shape)

    # Normalization 1
    X = BatchNormalization()(X_input)
    # a = X
    # b = X

    # Gated CNN 1
    a_sig = Conv1D(filters=128, kernel_size=(2), strides=1, kernel_regularizer=regularizers.l2(0.0005),
                   activation="sigmoid", padding="same")(X)
    a_relu = Conv1D(filters=128, kernel_size=(2), strides=1, kernel_regularizer=regularizers.l2(0.0005),
                    activation="relu", padding="same")(X)
    a = Multiply()([a_sig, a_relu])

    # Gated CNN 2
    b_sig = Conv1D(filters=128, kernel_size=(3), strides=1, kernel_regularizer=regularizers.l2(0.0005),
                   activation="sigmoid", padding="same")(X)
    b_relu = Conv1D(filters=128, kernel_size=(3), strides=1, kernel_regularizer=regularizers.l2(0.0005),
                    activation="relu", padding="same")(X)
    # b = Conv1D(filters=128, kernel_size=3, strides=1)(b)
    b = Multiply()([b_sig, b_relu])

    # Concatenate
    X = Concatenate()([a, b])

    # Normalization 2
    X = BatchNormalization()(X)

    # BidirectionalLSTM
    X = Bidirectional(LSTM(100, return_sequences=True))(X)

    X = GlobalMaxPooling1D()(X)
    # attention = Dense(1, activation='tanh')(X)
    # attention = Flatten()(attention)
    # attention = Activation('softmax')(attention)
    # attention = RepeatVector(200)(attention)
    # attention = Permute([2, 1])(attention)

    # s_r = Multiply()([X, attention])
    # s_r = Lambda(lambda  xin: K.sum(xin, axis=-2), output_shape=(200,))(s_r)

    X = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.0005))(X)

    X = Dropout(0.5)(X)

    X = Dense(1, kernel_regularizer=regularizers.l2(0.0005))(X)
    X = Activation("sigmoid")(X)
    model = Model(inputs=X_input, outputs=X)

    return model


'''
    STEP 3 : Train model
'''
num_samples = X_train.shape[0]


# https://stackoverflow.com/questions/39779710/setting-up-a-learningratescheduler-in-keras
# learning_scheduler = LearningRateScheduler()

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


# def KerasFocalLoss(target, input):
#     gamma = 2.
#     input = tf.cast(input, tf.float32)
#
#     max_val = K.clip(-input, 0, 1)
#     loss = input - input * target + max_val + K.log(K.exp(-max_val) + K.exp(-input - max_val))
#     invprobs = tf.log_sigmoid(-input * (target * 2.0 - 1.0))
#     loss = K.exp(invprobs * gamma) * loss
#
#     return K.mean(K.sum(loss, axis=1))
# def focal_loss(gamma=2., alpha=.25):
#   def focal_loss_fixed(y_true, y_pred):
#     pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
#     pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
#
#     pt_1 = K.clip(pt_1, 1e-3, .999)
#     pt_0 = K.clip(pt_0, 1e-3, .999)
#
#     return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))
# return focal_loss_fixed

generator2 = generate_data(X_train, Y_train, batch_size)

reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=8, verbose=1, mode='min')
terminate_on_nan = TerminateOnNaN()
model_checkpoint = ModelCheckpoint("cp_loss789_accuracy_98__v3", monitor='loss', save_best_only=True, mode='min')
early_stopping = EarlyStopping(monitor='loss', patience=8, mode='auto')

model = load_model("cp_loss789_accuracy_98__v3")
# model = malware_detection_model(input_shape=(1000, n_x))

learning_rate = 0.01
# decay_rate = learning_rate / epochs
momentum = 0.8


def step_decay(epoch):
    # initial_lrate = 0.1
    # drop = 0.5
    # epochs_drop = 10.0
    # lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    print("EPOCH NUMBER:", epoch)
    if epoch >= 13 and epoch <= 20:
        print("LR is 0.001")
        return 0.001
    elif epoch > 20:
        print("LR is 0.0001")
        return 0.0001
    print("LR is 0.01")
    return 0.01


lrate = LearningRateScheduler(step_decay)

opt = Adam(learning_rate=0.001, decay=1e-8)
# opt = SGD(learning_rate=0.0001, decay=1e-8, momentum=0.9)
# sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

# model.summary()

# model.fit(x=X_train, y=Y_train, epochs=10, batch_size=32, shuffle=True)
print("---Alright new thing###:", X_train.shape, X_val.shape, Y_train.shape, Y_val.shape)

model.fit_generator(
    generator2,
    steps_per_epoch=math.ceil(len(X_train) / batch_size),
    epochs=no_epochs,
    shuffle=True,
    verbose=1,
    initial_epoch=36,
    validation_data=(X_val, Y_val),
    callbacks=[early_stopping, terminate_on_nan, reduce_lr, model_checkpoint])

print("Training Done::Evaluating")
# print(model.evaluate(X_train[test], Y_train[test]))

# for train, test in kfolds.split(X_train, Y_train):
#     print('\nFold')
#     generator2 = generate_data(X_train[train], Y_train[train], batch_size)
#
#     reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=5, verbose=1, mode='min')
#     terminate_on_nan = TerminateOnNaN()
#     model_checkpoint = ModelCheckpoint("checkpoint.txt", monitor='loss', save_best_only=True, mode='min')
#
#     model = malware_detection_model(input_shape = (max_Tx, n_x))
#     opt = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
#     model.compile(loss='binary_crossentropy', optimizer=opt, metrics=["accuracy"])
#     #model.summary()
#     #model.fit(x=X_train, y=Y_train, epochs=10, batch_size=32, shuffle=True)
#
#     model.fit_generator(
#                 generator2,
#                 steps_per_epoch=math.ceil(len(X_train[train])/batch_size),
#                 epochs=no_epochs,
#                 shuffle=True,
#                 verbose=1,
#                 validation_data=(X_train[test], Y_train[test]),
#                 callbacks=[terminate_on_nan, model_checkpoint, reduce_lr])
#
#     print("Training Done::Evaluating")
#     print(model.evaluate(X_train[test], Y_train[test]))

'''
    STEP 4 : Prepare test samples
'''
X_test = []
for i in range(0, test_samples):
    data = np.load("data/test/test/" + str(i) + ".npy")
    # zero_mat = np.zeros((max_Tx, n_x))
    # zero_mat.put(i, data)
    # X_test.append(zero_mat)
    zero_mat = np.zeros((max_Tx, 102))
    zero_mat[:data.shape[0], :] = data
    X_test.append(zero_mat)

X_test = np.array(X_test)

print(X_test.shape)

'''
    STEP 5 : Predict on test
'''
pred = model.predict(X_test)

'''
    STEP 6 : Save data to csv
'''
print(pred.shape, pred)
pred = pd.DataFrame(data=pred, index=[i for i in range(pred.shape[0])], columns=["Predicted"])
pred.to_csv('attemp_v1.csv', index=True)

