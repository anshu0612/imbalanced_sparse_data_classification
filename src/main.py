import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.metrics import homogeneity_score
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

from keras.preprocessing import sequence
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional, Input, BatchNormalization, Conv1D, Multiply, Activation
X = []
y = []

max_len = 500
#336
batch_size = 50
train_samples = 400
# 30336
test_samples = 2
#10000

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('seaborn')


def load_data(dir):
	files = os.listdir(dir)
	X = np.zeros((len(files), 2000))
	print(X.shape)
	for index, file in enumerate(files):
		x_i = np.nan_to_num(np.load(os.path.join(dir, file)))
		x_i = x_i[0:min(50, x_i.shape[0])].reshape((-1,))
		X[int(file[:-4]), :x_i.shape[0]] = x_i
		if index is train_samples:
			break
	return X


X = load_data('data/train/train')
# y = pd.read_csv('/content/train_kaggle.csv')

# X_test = load_data('/content/test/test')


#X = load_data('/content/train/train')

#X = np.array(X)
df = pd.read_csv("data/train_kaggle.csv", usecols=["label"])
y = df[:train_samples]
y = y.values

# X = np.nan_to_num(np.array(X))
y = np.array(y)


x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)





# model = Sequential()
#
# data_input = Input(shape=(None, 40))
# normalize_input = BatchNormalization()(data_input)
#
# sig_conv = Conv1D(128, (2), activation='sigmoid', padding='same')(normalize_input)
# rel_conv = Conv1D(128, (2), activation='relu', padding='same')(normalize_input)
# mul_conv = Multiply()([sig_conv, rel_conv])
#
# lstm = Bidirectional(LSTM(64))(mul_conv)
# dense_1 = Dense(64, activation='relu')(lstm)
# dense_1 = Dropout(0.5)(dense_1)
# dense_2 = Dense(1)(dense_1)
# out = Activation('softmax')(dense_2)
# model = Model(input=data_input, output=out)
#
# model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
#
# print('Model', model.summary())
#
# model.fit(x_train, y_train,
# 		  batch_size=batch_size,
# 		  epochs = 100,
# 		  validation_data=[x_test, y_test])
