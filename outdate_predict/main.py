import random
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

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

block = pd.read_csv("../csv/block.csv")
method = pd.read_csv("../csv/method.csv")
top = pd.read_csv('block_method_top.csv')
block_method = pd.concat([block, method], axis=0)
# ------------------------ new version ------------------------
block_new = block.drop(
    ['lineNumOfOldCommentBylineNumOf"OldCCSet', 'lineNumOfOldCode', 'lineNumOfOldComment', 'lineNumOfChanged',
     'cmt2cd_sim_before', 'cmt2cd_sim_after'], axis=1)
method_new = method.drop(
    ['lineNumOfOldCommentBylineNumOf"OldCCSet', 'lineNumOfOldCode', 'lineNumOfOldComment', 'lineNumOfChanged',
     'cmt2cd_sim_before', 'cmt2cd_sim_after'], axis=1)
block_method_new = pd.concat([block_new, method_new], axis=0)
remove_word_anaysis = block_method_new.drop(['NNComment', 'VBComment', 'DTComment', 'INComment', 'JJComment',
                                             'RBComment', 'PRPComment', 'MDComment', 'LSComment', 'RPComment', 'NNCode',
                                             'VBCode', 'DTCode', 'INCode', 'JJCode',
                                             'RBCode', 'PRPCode', 'MDCode', 'LSCode', 'RPCode'], axis=1)
# code features
code_new = block_method_new.drop(
    ['TODOCount', 'FIXMECount', 'XXXCount', 'BUGCount',
     'VERSIONCount', 'FIXEDCount', 'NNComment', 'VBComment', 'DTComment', 'INComment', 'JJComment', 'RBComment',
     'PRPComment', 'MDComment', 'LSComment', 'RPComment', 'bothHavePairNumChange',
     'cmt2cd_sim_change', 'cmt2ch_sim_change', 'all_token_change_sim'
     ], axis=1)
# comment features
comment_new = block_method_new.drop(
    ['changeNum', 'attribute', 'methodDeclaration', 'methodRenaming', 'returnType', 'parameterDelete',
     'parameterInsert', 'parameterRenaming', 'parameterTypeChange', 'containReturn',
     'lineNumOfOldCodeBylineNumOfOldCCSet', 'changedLineByAllCodeLine',
     'ifInsert', 'ifUpdate', 'ifMove', 'ifDelete', 'forInsert', 'forUpdate', 'forMove', 'forDelete', 'foreachInsert',
     'foreachUpdate', 'foreachMove', 'foreachDelete', 'whileInsert', 'whileUpdate', 'whileMove', 'whileDelete',
     'catchInsert', 'catchUpdate', 'catchMove', 'catchDelete', 'tryInsert', 'tryUpdate', 'tryMove', 'tryDelete',
     'throwInsert', 'throwUpdate', 'throwMove', 'throwDelete', 'methodInvInsert', 'methodInvUpdate', 'methodInvMove',
     'methodInvDelete', 'assignInsert', 'assignUpdate', 'assignMove', 'assignDelete', 'varDecInsert', 'varDecUpdate',
     'varDecMove', 'varDecDelete', 'elseInsert', 'elseUpdate', 'elseMove', 'elseDelete', 'NNCode', 'VBCode', 'DTCode',
     'INCode', 'JJCode', 'RBCode', 'PRPCode', 'MDCode', 'LSCode', 'RPCode', 'bothHavePairNumChange',
     'cmt2cd_sim_change', 'cmt2ch_sim_change', 'all_token_change_sim'
     ], axis=1)
# relation features
relation_new = block_method_new.drop(
    ['changeNum', 'attribute', 'methodDeclaration', 'methodRenaming', 'returnType', 'parameterDelete',
     'parameterInsert', 'parameterRenaming', 'parameterTypeChange', 'containReturn',
     'lineNumOfOldCodeBylineNumOfOldCCSet',
     'TODOCount', 'FIXMECount', 'XXXCount', 'BUGCount', 'VERSIONCount', 'FIXEDCount',
     'changedLineByAllCodeLine', 'ifInsert', 'ifUpdate', 'ifMove', 'ifDelete', 'forInsert',
     'forUpdate', 'forMove', 'forDelete', 'foreachInsert', 'foreachUpdate', 'foreachMove', 'foreachDelete',
     'whileInsert', 'whileUpdate', 'whileMove', 'whileDelete', 'catchInsert', 'catchUpdate', 'catchMove', 'catchDelete',
     'tryInsert', 'tryUpdate', 'tryMove', 'tryDelete', 'throwInsert', 'throwUpdate', 'throwMove', 'throwDelete',
     'methodInvInsert', 'methodInvUpdate', 'methodInvMove', 'methodInvDelete', 'assignInsert', 'assignUpdate',
     'assignMove', 'assignDelete', 'varDecInsert', 'varDecUpdate', 'varDecMove', 'varDecDelete', 'elseInsert',
     'elseUpdate', 'elseMove', 'elseDelete', 'NNComment', 'VBComment', 'DTComment', 'INComment', 'JJComment',
     'RBComment', 'PRPComment', 'MDComment', 'LSComment', 'RPComment', 'NNCode', 'VBCode', 'DTCode', 'INCode', 'JJCode',
     'RBCode', 'PRPCode', 'MDCode', 'LSCode', 'RPCode'
     ], axis=1)
# ------------------------ prev_version ------------------------

block_prev = block.drop(
    ['changeNum', 'NNComment', 'VBComment', 'DTComment', 'INComment', 'JJComment', 'RBComment', 'PRPComment',
     'MDComment', 'LSComment', 'RPComment', 'NNCode', 'VBCode', 'DTCode', 'INCode', 'JJCode', 'RBCode', 'PRPCode',
     'MDCode', 'LSCode', 'RPCode', 'bothHavePairNumChange', 'all_token_change_sim'], axis=1)
method_prev = method.drop(
    ['changeNum', 'NNComment', 'VBComment', 'DTComment', 'INComment', 'JJComment', 'RBComment', 'PRPComment',
     'MDComment', 'LSComment', 'RPComment', 'NNCode', 'VBCode', 'DTCode', 'INCode', 'JJCode', 'RBCode', 'PRPCode',
     'MDCode', 'LSCode', 'RPCode', 'bothHavePairNumChange', 'all_token_change_sim'], axis=1)
block_method_prev = pd.concat([block_prev, method_prev], axis=0)
# code features
code_prev = block_method_prev.drop(
    ['lineNumOfOldCommentBylineNumOf"OldCCSet', 'lineNumOfOldComment', 'TODOCount', 'FIXMECount', 'XXXCount',
     'BUGCount',
     'VERSIONCount', 'FIXEDCount',
     'cmt2cd_sim_change', 'cmt2ch_sim_change'
     ], axis=1)
# comment features
comment_prev = block_method_prev.drop(
    ['attribute', 'methodDeclaration', 'methodRenaming', 'returnType', 'parameterDelete',
     'parameterInsert', 'parameterRenaming', 'parameterTypeChange', 'containReturn',
     'lineNumOfOldCodeBylineNumOfOldCCSet', 'lineNumOfOldCode', 'lineNumOfChanged', 'changedLineByAllCodeLine',
     'ifInsert', 'ifUpdate', 'ifMove', 'ifDelete', 'forInsert', 'forUpdate', 'forMove', 'forDelete', 'foreachInsert',
     'foreachUpdate', 'foreachMove', 'foreachDelete', 'whileInsert', 'whileUpdate', 'whileMove', 'whileDelete',
     'catchInsert', 'catchUpdate', 'catchMove', 'catchDelete', 'tryInsert', 'tryUpdate', 'tryMove', 'tryDelete',
     'throwInsert', 'throwUpdate', 'throwMove', 'throwDelete', 'methodInvInsert', 'methodInvUpdate', 'methodInvMove',
     'methodInvDelete', 'assignInsert', 'assignUpdate', 'assignMove', 'assignDelete', 'varDecInsert', 'varDecUpdate',
     'varDecMove', 'varDecDelete', 'elseInsert', 'elseUpdate', 'elseMove', 'elseDelete',
     'cmt2cd_sim_before', 'cmt2cd_sim_after', 'cmt2cd_sim_change', 'cmt2ch_sim_change'
     ], axis=1)
# relation features
relation_prev = block_method_prev.drop(
    ['attribute', 'methodDeclaration', 'methodRenaming', 'returnType', 'parameterDelete',
     'parameterInsert', 'parameterRenaming', 'parameterTypeChange', 'containReturn',
     'lineNumOfOldCodeBylineNumOfOldCCSet', 'lineNumOfOldCode', 'lineNumOfOldCommentBylineNumOf"OldCCSet',
     'lineNumOfOldComment', 'TODOCount', 'FIXMECount', 'XXXCount', 'BUGCount', 'VERSIONCount', 'FIXEDCount',
     'lineNumOfChanged', 'changedLineByAllCodeLine', 'ifInsert', 'ifUpdate', 'ifMove', 'ifDelete', 'forInsert',
     'forUpdate', 'forMove', 'forDelete', 'foreachInsert', 'foreachUpdate', 'foreachMove', 'foreachDelete',
     'whileInsert', 'whileUpdate', 'whileMove', 'whileDelete', 'catchInsert', 'catchUpdate', 'catchMove', 'catchDelete',
     'tryInsert', 'tryUpdate', 'tryMove', 'tryDelete', 'throwInsert', 'throwUpdate', 'throwMove', 'throwDelete',
     'methodInvInsert', 'methodInvUpdate', 'methodInvMove', 'methodInvDelete', 'assignInsert', 'assignUpdate',
     'assignMove', 'assignDelete', 'varDecInsert', 'varDecUpdate', 'varDecMove', 'varDecDelete', 'elseInsert',
     'elseUpdate', 'elseMove', 'elseDelete'
     ], axis=1)
# ------------ use which ------------
df = block_method_new

# df = df.drop(['NNComment', 'VBComment', 'DTComment', 'INComment', 'JJComment',
#               'RBComment', 'PRPComment', 'MDComment', 'LSComment', 'RPComment', 'NNCode', 'VBCode', 'DTCode', 'INCode',
#               'JJCode',
#               'RBCode', 'PRPCode', 'MDCode', 'LSCode', 'RPCode'], axis=1)

sns.heatmap(df.corr())
print('-----------df.RESULT.value_counts()---------------')
y = df.label
print(df.label.value_counts())
x = df.drop(['label'], axis=1)
num_fea = x.dtypes[x.dtypes != 'object'].index
x[num_fea] = x[num_fea].apply(lambda a: (a - a.mean()) / (a.std()))
x[num_fea] = x[num_fea].fillna(0)

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3, random_state=10)


def different_model():
    nbc = GaussianNB()
    svmc = SVC()
    lrc = LogisticRegression()
    dtc = DecisionTreeClassifier()
    rfc = RandomForestClassifier()
    xgb = XGBClassifier(objective='binary:logistic')

    nbc.fit(xtrain, ytrain)
    svmc.fit(xtrain, ytrain)
    lrc.fit(xtrain, ytrain)
    dtc.fit(xtrain, ytrain)
    rfc.fit(xtrain, ytrain)
    xgb.fit(xtrain, ytrain)

    nbc_pre = nbc.predict(xtest)
    svm_pre = svmc.predict(xtest)
    lrc_pre = lrc.predict(xtest)
    dtc_pre = dtc.predict(xtest)
    rfc_pre = rfc.predict(xtest)
    xgb_pre = xgb.predict(xtest)

    print('-------------------- 默认参数 --------------------')
    print('precision, f1, recall')
    print('NaiveBayes:\t', precision_score(ytest, nbc_pre), f1_score(ytest, nbc_pre), recall_score(ytest, nbc_pre))
    # print('KNN:\t', precision_score(ytest, knnc_pre), f1_score(ytest, knnc_pre), recall_score(ytest, knnc_pre))
    print('SVM:\t', precision_score(ytest, svm_pre), f1_score(ytest, svm_pre), recall_score(ytest, svm_pre))
    print('LogisticRegression:\t', precision_score(ytest, lrc_pre), f1_score(ytest, lrc_pre),
          recall_score(ytest, lrc_pre))
    print('DecisionTree:\t', precision_score(ytest, dtc_pre), f1_score(ytest, dtc_pre), recall_score(ytest, dtc_pre))
    print('RandomForest:\t', precision_score(ytest, rfc_pre), f1_score(ytest, rfc_pre), recall_score(ytest, rfc_pre))
    print('XGBoost:\t', precision_score(ytest, xgb_pre), f1_score(ytest, xgb_pre), recall_score(ytest, xgb_pre))

    

    print('-------------------- GaussianNB网格搜索 --------------------')
    param_grid = {'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]}
    gnb = GaussianNB()
    cv = 10
    scoring = 'accuracy'
    grid_search = GridSearchCV(gnb, param_grid, cv=cv, scoring=scoring)
    grid_search.fit(xtrain, ytrain)
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    print("Best parameters:", best_params)
    print("Best score:", grid_search.best_score_)
    print("Best model:", best_model)

    print('-------------------- LogisticRegression网格搜索 --------------------')
    param_grid = {
        'penalty': ['l1', 'l2'],
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'solver': ['liblinear', 'saga']
    }
    lr = LogisticRegression()
    cv = 10
    scoring = 'accuracy'
    grid_search = GridSearchCV(lr, param_grid, cv=cv, scoring=scoring)
    grid_search.fit(xtrain, ytrain)
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    print("Best parameters:", best_params)
    print("Best score:", grid_search.best_score_)
    print("Best model:", best_model)

    print('-------------------- Decision Tree网格搜索 --------------------')
    param_grid = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 5, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    dt = DecisionTreeClassifier()
    cv = 10
    scoring = 'accuracy'
    grid_search = GridSearchCV(dt, param_grid, cv=cv, scoring=scoring)
    grid_search.fit(xtrain, ytrain)
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    print("Best parameters:", best_params)
    print("Best score:", grid_search.best_score_)
    print("Best model:", best_model)


    print('-------------------- XGBoost 网格搜索 --------------------')
    param_grid = {
        'learning_rate': [0.01, 0.1, 1],
        'max_depth': [3, 5, 7],
        'n_estimators': [50, 100, 200],
        'subsample': [0.5, 0.7, 1.0],
        'colsample_bytree': [0.5, 0.7, 1.0],
    }
    xgb_clf = XGBClassifier(objective='binary:logistic')
    cv = 10
    scoring = 'accuracy'
    grid_search = GridSearchCV(xgb_clf, param_grid, cv=cv, scoring=scoring)
    grid_search.fit(xtrain, ytrain)
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    print("Best parameters:", best_params)
    print("Best score:", grid_search.best_score_)
    print("Best model:", best_model)

    print('-------------------- Random Forest 网格搜索 --------------------')
    param_grid = {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [None, 5, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    rfc = RandomForestClassifier()
    cv = 10
    scoring = 'accuracy'
    grid_search = GridSearchCV(rfc, param_grid, cv=cv, scoring=scoring)
    grid_search.fit(xtrain, ytrain)
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    print("Best parameters:", best_params)
    print("Best score:", grid_search.best_score_)
    print("Best model:", best_model)
    
    print('-------------------- SVM网格搜索 --------------------')
    param_grid = {'C': [0.1, 1, 10, 100],
                  'gamma': [0.1, 1, 10, 100],
                  'kernel': ['linear', 'rbf']}
    svc = SVC()
    cv = 10
    scoring = 'accuracy'
    grid_search = GridSearchCV(svc, param_grid, cv=cv, scoring=scoring)
    grid_search.fit(xtrain, ytrain)
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    print("Best parameters:", best_params)
    print("Best score:", grid_search.best_score_)
    print("Best model:", best_model)
    
    
def random_forest_pred():
    rfc = RandomForestClassifier()
    rfc = rfc.fit(xtrain, ytrain)
    print('------------feature_importances_--------------')
    importances = rfc.feature_importances_
    std = np.std([tree.feature_importances_ for tree in rfc.estimators_], axis=0)
    indices = np.argsort(importances)[::-1]
    print('feature ranking')
    for f in range(min(50, xtrain.shape[1])):
        print('%2d) %-*s %f' % (f + 1, 30, xtrain.columns[indices[f]], importances[indices[f]]))
    plt.figure()
    plt.title('feature importances')
    plt.bar(range(xtrain.shape[1]), importances[indices], color='r', yerr=std[indices], align='center')
    plt.xticks(range(xtrain.shape[1]), indices)
    plt.xlim([-1, xtrain.shape[1]])
    plt.show()
    predicted = rfc.predict(xtest)

    cm2 = confusion_matrix(ytest, predicted)
    print(cm2)
    print('acc:', accuracy_score(ytest, predicted))
    print("过时注释:")
    print('Recall:', recall_score(ytest, predicted))
    print('F1-score:', f1_score(ytest, predicted))
    print('Precision score:', precision_score(ytest, predicted))


different_model()
# random_forest_pred()
