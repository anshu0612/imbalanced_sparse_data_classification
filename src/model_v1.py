import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

from xgboost import XGBRegressor
# import lightgbm

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
batch_size = 50
train_samples = 30336
# 30336
test_samples = 10000
#10000

X_t = []
for i in range(0, train_samples):
    data = np.load("data/train/train/" + str(i) + ".npy")

    zero_mat = np.zeros((max_len, 40))
    zero_mat[:data.shape[0], :] = data

    for feature in range(40):
        average_value = np.nanmean(zero_mat[:feature][np.nan_to_num(zero_mat[:feature]) != 0])
        zero_mat[:feature] = np.nan_to_num(zero_mat[:feature], average_value)
    zero_mat = zero_mat.reshape((-1,))
    X_t.append(zero_mat)

X = np.nan_to_num(np.array(X_t))
df = pd.read_csv("data/train_kaggle.csv", usecols=["label"])
Y_t = df[:train_samples]
y = Y_t.values

X_train, X_val, Y_train, Y_val = train_test_split(X, y, shuffle=True, test_size=0.20)
print(X_train.shape)
variance = VarianceThreshold(threshold=0.1)
X_train_new = variance.fit_transform(X_train)
X_test_new = variance.transform(X_val)
print(X_train_new.shape)

pca = PCA(n_components=100)
X_train_new = pca.fit_transform(X_train_new)
X_val_new = pca.transform(X_test_new)
print(X_train_new.shape)

# # print("SVM")
# # svm = SVC(kernel="linear", C=0.025, probability=True)
# # svm.fit(X_train_new, Y_train)
# # y_pred = svm.predict_proba(X_val_new)[:,1]
# # classification_evaluation(Y_val, y_pred)
# #
#print("====DecisionTreeClassifier====")
#dtc = DecisionTreeRegressor(max_depth=8,  min_samples_leaf=2)
#dtc.fit(X_train_new, Y_train)
#y_pred = dtc.predict(X_val_new)
#classification_evaluation(Y_val, y_pred)
#
# print("RandomForestClassifier")
# rmc = RandomForestRegressor(n_estimators=200, min_samples_leaf=2, max_depth=8)
# rmc.fit(X_train_new, Y_train)
# y_pred = rmc.predict(X_val_new)
# classification_evaluation(Y_val, y_pred)
#
#
#print("======GradientBoostingRegressor====")
#gbc = GradientBoostingRegressor(n_estimators=500, min_samples_leaf=2, max_depth=8)
#gbc.fit(X_train_new, Y_train)
#y_pred = gbc.predict(X_val_new)
#classification_evaluation(Y_val, y_pred)

print("=======XGBRegressor=====")
xgb_model = XGBRegressor(objective="binary:logistic", max_depth=8, min_samples_leaf=2, n_estimators=300, random_state=42)
xgb_model.fit(X_train_new, Y_train)
y_pred = xgb_model.predict(X_val_new)
classification_evaluation(Y_val, y_pred)

'''
    STEP 4 : Prepare test samples
'''
X_test = []
for i in range(0, test_samples):
    data = np.load("data/test/test/" + str(i) + ".npy")
    zero_mat = np.zeros((max_len, 40))
    zero_mat[:data.shape[0], :] = data
    for feature in range(40):
        average_value = np.nanmean(zero_mat[:feature][np.nan_to_num(zero_mat[:feature]) != 0])
        zero_mat[:feature] = np.nan_to_num(zero_mat[:feature], average_value)
    zero_mat = zero_mat.reshape((-1,))
    X_test.append(zero_mat)

X_test = np.nan_to_num(np.array(X_test))
print(X_test.shape)
'''
    STEP 5 : Predict on test
'''
X_test = variance.fit_transform(X_test)
X_test = pca.fit_transform(X_test)
'''
    STEP 6 : Save data to csv
'''
#XGBOOST
pred = xgb_model.predict(X_test)
print(pred.shape, pred)
pred = pd.DataFrame(data=pred, index=[i for i in range(pred.shape[0])], columns=["Predicted"])
pred.index.name = 'Id'
pred.to_csv('xgboost.csv', index=True)

#GradietBoosting
#pred = gbc.predict(X_test)
#print(pred.shape, pred)
#pred = pd.DataFrame(data=pred, index=[i for i in range(pred.shape[0])], columns=["Predicted"])
#pred.index.name = 'Id'
#pred.to_csv('gbc.csv', index=True)

#print("====DecisionTreeClassifier====")
#pred = dtc.predict(X_test)
#pred = pd.DataFrame(data=pred, index=[i for i in range(pred.shape[0])], columns=["Predicted"])
#pred.index.name = 'Id'
#pred.to_csv('dtc.csv', index=True)
