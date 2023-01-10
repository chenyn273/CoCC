import random
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import scipy
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
import random
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import roc_curve, auc, roc_auc_score, precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.utils import shuffle
from xgboost import XGBClassifier
import seaborn as sns

import matplotlib.pyplot as plt

py = pd.read_csv('python.csv')

df = py
y = df.label
x = df.drop(['label'], axis=1)

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3, random_state=10)

def different_model():
    nbc = GaussianNB()
    knnc = KNeighborsClassifier()
    svmc = SVC()
    lrc = LogisticRegression()
    dtc = DecisionTreeClassifier()
    rfc = RandomForestClassifier()
    xgb = XGBClassifier()

    nbc.fit(xtrain, ytrain)
    knnc.fit(xtrain, ytrain)
    svmc.fit(xtrain, ytrain)
    lrc.fit(xtrain, ytrain)
    dtc.fit(xtrain, ytrain)
    rfc.fit(xtrain, ytrain)
    xgb.fit(xtrain, ytrain)

    nbc_pre = nbc.predict(xtest)
    knnc_pre = knnc.predict(xtest)
    svm_pre = svmc.predict(xtest)
    lrc_pre = lrc.predict(xtest)
    dtc_pre = dtc.predict(xtest)
    rfc_pre = rfc.predict(xtest)
    xgb_pre = xgb.predict(xtest)

    print('-------------------- 默认参数 --------------------')
    print('precision, f1, recall')
    print('NaiveBayes:\t', precision_score(ytest, nbc_pre), f1_score(ytest, nbc_pre), recall_score(ytest, nbc_pre))
    print('KNN:\t', precision_score(ytest, knnc_pre), f1_score(ytest, knnc_pre), recall_score(ytest, knnc_pre))
    print('SVM:\t', precision_score(ytest, svm_pre), f1_score(ytest, svm_pre), recall_score(ytest, svm_pre))
    print('LogisticRegression:\t', precision_score(ytest, lrc_pre), f1_score(ytest, lrc_pre),
          recall_score(ytest, lrc_pre))
    print('DecisionTree:\t', precision_score(ytest, dtc_pre), f1_score(ytest, dtc_pre), recall_score(ytest, dtc_pre))
    print('RandomForest:\t', precision_score(ytest, rfc_pre), f1_score(ytest, rfc_pre), recall_score(ytest, rfc_pre))
    print('XGBoost:\t', precision_score(ytest, xgb_pre), f1_score(ytest, xgb_pre), recall_score(ytest, xgb_pre))

    print('-------------------- SVM网格搜索 --------------------')
    model = SVC()
    params = [
        {'kernel': ['linear'], 'C': [0.05, 0.1, 0.5, 0.75, 0.9, 1, 2, 10, 50, 100, 1000]},
        {'kernel': ['poly'], 'C': [0.05, 0.1, 0.5, 0.75, 0.9, 1, 2, 10, 50, 100, 1000], 'degree': [2, 3]},
        {'kernel': ['rbf'], 'C': [0.05, 0.1, 0.5, 0.75, 0.9, 1, 2, 10, 50, 100, 1000],
         'gamma': [1, 0.5, 0.1, 0.01, 0.001]}]
    model = GridSearchCV(estimator=model, param_grid=params, scoring='precision', cv=10)  # precision recall f1
    model.fit(xtrain, ytrain)
    print("SVM_precsion模型的最优参数：", model.best_params_)
    svm_best_model = model.best_estimator_
    pred = svm_best_model.predict(xtest)
    prec, f1, recall = precision_score(ytest, pred), f1_score(ytest, pred), recall_score(ytest, pred)
    print('SVM precision on test:', prec)
    print('SVM f1 on test:', f1)
    print('SVM recall on test:', recall)

    model = SVC()
    params = [
        {'kernel': ['linear'], 'C': [0.05, 0.1, 0.5, 0.75, 0.9, 1, 2, 10, 50, 100, 1000]},
        {'kernel': ['poly'], 'C': [0.05, 0.1, 0.5, 0.75, 0.9, 1, 2, 10, 50, 100, 1000], 'degree': [2, 3]},
        {'kernel': ['rbf'], 'C': [0.05, 0.1, 0.5, 0.75, 0.9, 1, 2, 10, 50, 100, 1000],
         'gamma': [1, 0.5, 0.1, 0.01, 0.001]}]
    model = GridSearchCV(estimator=model, param_grid=params, scoring='accuracy', cv=10)  # precision recall f1
    model.fit(xtrain, ytrain)
    print("SVM_accuracy模型的最优参数：", model.best_params_)
    svm_best_model = model.best_estimator_
    pred = svm_best_model.predict(xtest)
    prec, f1, recall = precision_score(ytest, pred), f1_score(ytest, pred), recall_score(ytest, pred)
    print('SVM precision on test:', prec)
    print('SVM f1 on test:', f1)
    print('SVM recall on test:', recall)

    model = SVC()
    params = [
        {'kernel': ['linear'], 'C': [0.05, 0.1, 0.5, 0.75, 0.9, 1, 2, 10, 50, 100, 1000]},
        {'kernel': ['poly'], 'C': [0.05, 0.1, 0.5, 0.75, 0.9, 1, 2, 10, 50, 100, 1000], 'degree': [2, 3]},
        {'kernel': ['rbf'], 'C': [0.05, 0.1, 0.5, 0.75, 0.9, 1, 2, 10, 50, 100, 1000],
         'gamma': [1, 0.5, 0.1, 0.01, 0.001]}]
    model = GridSearchCV(estimator=model, param_grid=params, scoring='f1', cv=10)  # precision recall f1
    model.fit(xtrain, ytrain)
    print("SVM_f1模型的最优参数：", model.best_params_)
    svm_best_model = model.best_estimator_
    pred = svm_best_model.predict(xtest)
    prec, f1, recall = precision_score(ytest, pred), f1_score(ytest, pred), recall_score(ytest, pred)
    print('SVM precision on test:', prec)
    print('SVM f1 on test:', f1)
    print('SVM recall on test:', recall)

    model = SVC()
    params = [
        {'kernel': ['linear'], 'C': [0.05, 0.1, 0.5, 0.75, 0.9, 1, 2, 10, 50, 100, 1000]},
        {'kernel': ['poly'], 'C': [0.05, 0.1, 0.5, 0.75, 0.9, 1, 2, 10, 50, 100, 1000], 'degree': [2, 3]},
        {'kernel': ['rbf'], 'C': [0.05, 0.1, 0.5, 0.75, 0.9, 1, 2, 10, 50, 100, 1000],
         'gamma': [1, 0.5, 0.1, 0.01, 0.001]}]
    model = GridSearchCV(estimator=model, param_grid=params, scoring='roc_auc', cv=10)  # precision recall f1
    model.fit(xtrain, ytrain)
    print("SVM_roc_auc模型的最优参数：", model.best_params_)
    svm_best_model = model.best_estimator_
    pred = svm_best_model.predict(xtest)
    prec, f1, recall = precision_score(ytest, pred), f1_score(ytest, pred), recall_score(ytest, pred)
    print('SVM precision on test:', prec)
    print('SVM f1 on test:', f1)
    print('SVM recall on test:', recall)

    print('-------------------- LogisticRegression网格搜索 --------------------')
    lr_model = LogisticRegression()
    lr_params = {
        'C': [0.05, 0.1, 0.5, 0.75, 0.9, 1, 2, 10, 50, 100, 1000],
        'penalty': ['l1', 'l2'],
    }
    lr_model = GridSearchCV(estimator=lr_model, param_grid=lr_params, scoring='precision', cv=10)  # precision recall f1
    lr_model.fit(xtrain, ytrain)
    print("LR_precision模型的最优参数：", lr_model.best_params_)
    lr_best_model = lr_model.best_estimator_
    lr_pred = lr_best_model.predict(xtest)
    prec, f1, recall = precision_score(ytest, lr_pred), f1_score(ytest, lr_pred), recall_score(ytest, lr_pred)
    print('LR precision on test:', prec)
    print('LR f1 on test:', f1)
    print('LR recall on test:', recall)

    lr_model = LogisticRegression()
    lr_params = {
        'C': [0.05, 0.1, 0.5, 0.75, 0.9, 1, 2, 10, 50, 100, 1000],
        'penalty': ['l1', 'l2'],
    }
    lr_model = GridSearchCV(estimator=lr_model, param_grid=lr_params, scoring='accuracy', cv=10)  # precision recall f1
    lr_model.fit(xtrain, ytrain)
    print("LR_accuracy模型的最优参数：", lr_model.best_params_)
    lr_best_model = lr_model.best_estimator_
    lr_pred = lr_best_model.predict(xtest)
    prec, f1, recall = precision_score(ytest, lr_pred), f1_score(ytest, lr_pred), recall_score(ytest, lr_pred)
    print('LR precision on test:', prec)
    print('LR f1 on test:', f1)
    print('LR recall on test:', recall)

    lr_model = LogisticRegression()
    lr_params = {
        'C': [0.05, 0.1, 0.5, 0.75, 0.9, 1, 2, 10, 50, 100, 1000],
        'penalty': ['l1', 'l2'],
    }
    lr_model = GridSearchCV(estimator=lr_model, param_grid=lr_params, scoring='f1', cv=10)  # precision recall f1
    lr_model.fit(xtrain, ytrain)
    print("LR_f1模型的最优参数：", lr_model.best_params_)
    lr_best_model = lr_model.best_estimator_
    lr_pred = lr_best_model.predict(xtest)
    prec, f1, recall = precision_score(ytest, lr_pred), f1_score(ytest, lr_pred), recall_score(ytest, lr_pred)
    print('LR precision on test:', prec)
    print('LR f1 on test:', f1)
    print('LR recall on test:', recall)

    lr_model = LogisticRegression()
    lr_params = {
        'C': [0.05, 0.1, 0.5, 0.75, 0.9, 1, 2, 10, 50, 100, 1000],
        'penalty': ['l1', 'l2'],
    }
    lr_model = GridSearchCV(estimator=lr_model, param_grid=lr_params, scoring='roc_auc', cv=10)  # precision recall f1
    lr_model.fit(xtrain, ytrain)
    print("LR_roc_auc模型的最优参数：", lr_model.best_params_)
    lr_best_model = lr_model.best_estimator_
    lr_pred = lr_best_model.predict(xtest)
    prec, f1, recall = precision_score(ytest, lr_pred), f1_score(ytest, lr_pred), recall_score(ytest, lr_pred)
    print('LR precision on test:', prec)
    print('LR f1 on test:', f1)
    print('LR recall on test:', recall)

    print('-------------------- RandomForest网格搜索 --------------------')
    lr_model = RandomForestClassifier()
    lr_params = {
        'max_depth': [3, 5, 10, None],
        'n_estimators': [10, 50, 100, 200, 300],
        'max_features': list(range(6, 50)),
        'min_samples_leaf': [1, 2, 3],
        'min_samples_split': [2, 3],
        'criterion': ['gini', 'entropy']
    }
    lr_model = GridSearchCV(estimator=lr_model, param_grid=lr_params, scoring='precision', cv=10)  # precision recall f1
    lr_model.fit(xtrain, ytrain)
    print("RandomForest_precision模型的最优参数：", lr_model.best_params_)
    lr_best_model = lr_model.best_estimator_
    lr_pred = lr_best_model.predict(xtest)
    prec, f1, recall = precision_score(ytest, lr_pred), f1_score(ytest, lr_pred), recall_score(ytest, lr_pred)
    print('RF precision on test:', prec)
    print('RF f1 on test:', f1)
    print('RF recall on test:', recall)

    lr_model = RandomForestClassifier()
    lr_params = {
        'max_depth': [3, 5, 10, None],
        'n_estimators': [10, 50, 100, 200, 300],
        'max_features': list(range(6, 50)),
        'min_samples_leaf': [1, 2, 3],
        'min_samples_split': [2, 3],
        'criterion': ['gini', 'entropy']
    }
    lr_model = GridSearchCV(estimator=lr_model, param_grid=lr_params, scoring='accuracy', cv=10)  # precision recall f1
    lr_model.fit(xtrain, ytrain)
    print("RandomForest_accruracy模型的最优参数：", lr_model.best_params_)
    lr_best_model = lr_model.best_estimator_
    lr_pred = lr_best_model.predict(xtest)
    prec, f1, recall = precision_score(ytest, lr_pred), f1_score(ytest, lr_pred), recall_score(ytest, lr_pred)
    print('RF precision on test:', prec)
    print('RF f1 on test:', f1)
    print('RF recall on test:', recall)

    lr_model = RandomForestClassifier()
    lr_params = {
        'max_depth': [3, 5, 10, None],
        'n_estimators': [10, 50, 100, 200, 300],
        'max_features': list(range(6, 50)),
        'min_samples_leaf': [1, 2, 3],
        'min_samples_split': [2, 3],
        'criterion': ['gini', 'entropy']
    }
    lr_model = GridSearchCV(estimator=lr_model, param_grid=lr_params, scoring='f1', cv=10)  # precision recall f1
    lr_model.fit(xtrain, ytrain)
    print("RandomForest_f1模型的最优参数：", lr_model.best_params_)
    lr_best_model = lr_model.best_estimator_
    lr_pred = lr_best_model.predict(xtest)
    prec, f1, recall = precision_score(ytest, lr_pred), f1_score(ytest, lr_pred), recall_score(ytest, lr_pred)
    print('RF precision on test:', prec)
    print('RF f1 on test:', f1)
    print('RF recall on test:', recall)

    lr_model = RandomForestClassifier()
    lr_params = {
        'max_depth': [3, 5, 10, None],
        'n_estimators': [10, 50, 100, 200, 300],
        'max_features': list(range(6, 50)),
        'min_samples_leaf': [1, 2, 3],
        'min_samples_split': [2, 3],
        'criterion': ['gini', 'entropy']
    }
    lr_model = GridSearchCV(estimator=lr_model, param_grid=lr_params, scoring='roc_auc', cv=10)  # precision recall f1
    lr_model.fit(xtrain, ytrain)
    print("RandomForest_roc_auc模型的最优参数：", lr_model.best_params_)
    lr_best_model = lr_model.best_estimator_
    lr_pred = lr_best_model.predict(xtest)
    prec, f1, recall = precision_score(ytest, lr_pred), f1_score(ytest, lr_pred), recall_score(ytest, lr_pred)
    print('RF precision on test:', prec)
    print('RF f1 on test:', f1)
    print('RF recall on test:', recall)

def random_forest_pred():
    rfc = XGBClassifier()
    rfc = rfc.fit(xtrain, ytrain)

    predicted = rfc.predict(xtest)
    print('Recall:', recall_score(ytest, predicted))
    print('F1-score:', f1_score(ytest, predicted))
    print('Precision score:', precision_score(ytest, predicted))


different_model()
# random_forest_pred()
