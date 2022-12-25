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

import matplotlib.pyplot as plt

ejbca = pd.read_csv("/Users/chenyn/chenyn's/研究生/DataSet/My dect/RQ4/ejbca.csv")
freecol = pd.read_csv("/Users/chenyn/chenyn's/研究生/DataSet/My dect/RQ4/freecol.csv")
opennms = pd.read_csv("/Users/chenyn/chenyn's/研究生/DataSet/My dect/RQ4/opennms.csv")
to_dect = pd.concat([ejbca, freecol, opennms], axis=0)

block = pd.read_csv("/Users/chenyn/chenyn's/研究生/DataSet/My dect/csv/block.csv")
method = pd.read_csv("/Users/chenyn/chenyn's/研究生/DataSet/My dect/csv/method.csv")
block_method = pd.concat([block, method], axis=0)
# ------------------------ new version ------------------------
to_dect = to_dect.drop(
    ['lineNumOfOldCode', 'lineNumOfOldComment', 'lineNumOfChanged', 'cmt2cd_sim_before', 'cmt2cd_sim_after'], axis=1)
block_new = block.drop(
    ['lineNumOfOldCode', 'lineNumOfOldComment', 'lineNumOfChanged', 'cmt2cd_sim_before', 'cmt2cd_sim_after'], axis=1)
method_new = method.drop(
    ['lineNumOfOldCode', 'lineNumOfOldComment', 'lineNumOfChanged', 'cmt2cd_sim_before', 'cmt2cd_sim_after'], axis=1)
block_method_new = pd.concat([block_new, method_new], axis=0)
# code features
code_new = block_method_new.drop(
    ['lineNumOfOldCommentBylineNumOf"OldCCSet', 'TODOCount', 'FIXMECount', 'XXXCount', 'BUGCount',
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
     'lineNumOfOldCodeBylineNumOfOldCCSet', 'lineNumOfOldCommentBylineNumOf"OldCCSet',
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
print('-----------df.RESULT.value_counts()---------------')
to_dect_y = to_dect.label
y = df.label
print(df.label.value_counts())
to_dect_x = to_dect.drop(['label', 'filename'], axis=1)
x = df.drop(['label'], axis=1)

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3, random_state=10)


def detect_newly():
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
    to_dect_pre = rfc.predict(to_dect_x)
    print('to_dect:\t', precision_score(to_dect_y, to_dect_pre), f1_score(to_dect_y, to_dect_pre),
          recall_score(to_dect_y, to_dect_pre))
    print(sum(to_dect_pre == 1))
    print(len(to_dect_pre))
    print(to_dect.iloc[to_dect_pre == 1])
    print(len(to_dect.iloc[to_dect_pre == 1]))
    with open('detected_outdated.txt', 'a') as f:
        f.write(str(to_dect.iloc[to_dect_pre == 1].filename))


detect_newly()

