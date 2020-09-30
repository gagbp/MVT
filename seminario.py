import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import sklearn

# Import necessary modules
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn import model_selection
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import LeavePOut
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import StratifiedKFold

dat = pd.read_csv('heart_failure_clinical_records_dataset.csv') 
x1 = dat.drop('DEATH_EVENT', axis=1).values ## variavel alvo stars
y1 = dat['DEATH_EVENT'].values 

print(dat.shape)
#print(dat.describe().transpose())
print(dat.transpose().transpose())


print("\nValidações:\n  1-Holdout Validation\n  2-K-fold Cross-Validation\n  3-Stratified K-fold Cross-Validation\n  4-LOOCV\n  5-Repeated Random Test-Train Splits\n\nDigite o código da validação desejada: ")
N = int(input())

if N == 1:
  X_train, X_test, Y_train, Y_test = model_selection.train_test_split(x1, y1, test_size=0.30, random_state=None)
  model = LogisticRegression(solver='liblinear')
  model.fit(X_train, Y_train)
  result = model.score(X_test, Y_test)
  print("Holdout Validation Approach : Accuracy: %.2f%%" % (result*100.0))
elif N == 2:
  kfold = model_selection.KFold(n_splits=20, random_state=None)
  model_kfold = LogisticRegression(solver='liblinear')
  results_kfold = model_selection.cross_val_score(model_kfold, x1, y1, cv=kfold)
  print("K-fold Cross-Validation : Accuracy: %.2f%%" % (results_kfold.mean()*100.0))
elif N == 3:
  skfold = StratifiedKFold(n_splits=10, random_state=100,shuffle=True)
  model_skfold = LogisticRegression(solver='liblinear')
  results_skfold = model_selection.cross_val_score(model_skfold, x1, y1, cv=skfold)
  print("Stratified K-fold Cross-Validation : Accuracy: %.2f%%" % (results_skfold.mean()*100.0))
elif N == 4:
  loocv = model_selection.LeaveOneOut()
  model_loocv = LogisticRegression(solver='liblinear')
  results_loocv = model_selection.cross_val_score(model_loocv, x1, y1, cv=loocv)
  print("LOOCV : Accuracy: %.2f%%" % (results_loocv.mean()*100.0))
elif N == 5:
  kfold2 = model_selection.ShuffleSplit(n_splits=10, test_size=0.30, random_state=100)
  model_shufflecv = LogisticRegression(solver='liblinear')
  results_4 = model_selection.cross_val_score(model_shufflecv, x1, y1, cv=kfold2)
  print("Repeated Random Test-Train Splits : Accuracy: %.2f%% (%.2f%%)" % (results_4.mean()*100.0,results_4.std()*100.0))
else:
  print("Código inválido")
