import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

from xgboost import XGBClassifier
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

max_len = 350
#336
batch_size = 50
train_samples = 500
# 30336
test_samples = 100
#10000

X_t = []
for i in range(0, train_samples):
    data = np.nan_to_num(np.load("data/train/train/" + str(i) + ".npy"))
    zero_mat = np.zeros((max_len, 40))
    zero_mat[:data.shape[0], :] = data
    zero_mat = zero_mat.reshape((-1,))
    X_t.append(zero_mat)

X = np.array(X_t)
df = pd.read_csv("data/train_kaggle.csv", usecols=["label"])
Y_t = df[:train_samples]
y = Y_t.values

X_train, X_val, Y_train, Y_val = train_test_split(X, y, shuffle=True, test_size=0.20)

pca = PCA(n_components=100)
X_train_new = pca.fit_transform(X_train)
X_val_new = pca.transform(X_val)
#
# print("SVM")
# svm = SVC(kernel="linear", C=0.025, probability=True)
# svm.fit(X_train_new, Y_train)
# y_pred = svm.predict_proba(X_val_new)[:,1]
# classification_evaluation(Y_val, y_pred)
#
# print("DecisionTreeClassifier")
# dtc = DecisionTreeClassifier()
# dtc.fit(X_train_new, Y_train)
# y_pred = dtc.predict_proba(X_val_new)[:,1]
# classification_evaluation(Y_val, y_pred)
#
# print("RandomForestClassifier")
# rmc = RandomForestClassifier(n_estimators=20)
# rmc.fit(X_train_new, Y_train)
# y_pred = rmc.predict_proba(X_val_new)[:,1]
# classification_evaluation(Y_val, y_pred)
#
#
# print("GradientBoostingClassifier")
# gbc = GradientBoostingClassifier(n_estimators=20)
# gbc.fit(X_train_new, Y_train)
# y_pred = gbc.predict_proba(X_val_new)[:,1]
# classification_evaluation(Y_val, y_pred)

print("XGBClassifier")
xgb_model = XGBClassifier(objective="binary:logistic", random_state=42)
xgb_model.fit(X_train_new, Y_train)
y_pred = xgb_model.predict_proba(X_val_new)[:,1]
classification_evaluation(Y_val, y_pred)

'''
    STEP 4 : Prepare test samples
'''
X_test = []
for i in range(0, test_samples):
    data = np.nan_to_num(np.load("data/test/test/" + str(i) + ".npy"))
    zero_mat = np.zeros((max_len, 40))
    zero_mat[:data.shape[0], :] = data
    zero_mat = zero_mat.reshape((-1,))
    X_test.append(zero_mat)

X_test = np.array(X_test)

print(X_test.shape)

'''
    STEP 5 : Predict on test
'''
pca = PCA(n_components=100)
X_test = pca.fit_transform(X_test)
pred = xgb_model.predict(X_test)

'''
    STEP 6 : Save data to csv
'''
print(pred.shape, pred)
pred = pd.DataFrame(data=pred, index=[i for i in range(pred.shape[0])], columns=["Predicted"])
pred.index.name = 'Id'
pred.to_csv('attemp_v1.csv', index=True)