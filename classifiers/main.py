import numpy as np
import scipy
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
from sklearn import linear_model
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn import naive_bayes
from sklearn import tree
from sklearn import svm
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.calibration import CalibrationDisplay
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.svm import LinearSVC

block = pd.read_csv("/Users/chenyn/chenyn's/研究生/DataSet/My dect/csv/block.csv")
method = pd.read_csv("/Users/chenyn/chenyn's/研究生/DataSet/My dect/csv/method.csv")
block_method = pd.concat([block, method], axis=0)
# ------------------------ new version ------------------------
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
y = df.label
x = df.drop('label', axis=1)

seed = 8
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3, random_state=seed)

rfc = RandomForestClassifier(n_estimators=100, criterion='gini', max_features='sqrt')

class NaivelyCalibratedLinearSVC(LinearSVC):
    """LinearSVC with `predict_proba` method that naively scales
    `decision_function` output."""

    def fit(self, X, y):
        super().fit(X, y)
        df = self.decision_function(X)
        self.df_min_ = df.min()
        self.df_max_ = df.max()

    def predict_proba(self, X):
        """Min-max scale output of `decision_function` to [0,1]."""
        df = self.decision_function(X)
        calibrated_df = (df - self.df_min_) / (self.df_max_ - self.df_min_)
        proba_pos_class = np.clip(calibrated_df, 0, 1)
        proba_neg_class = 1 - proba_pos_class
        proba = np.c_[proba_neg_class, proba_pos_class]
        return proba


svc = NaivelyCalibratedLinearSVC(C=1.0, dual=False, max_iter=1000)
gnb = naive_bayes.GaussianNB()
lr = linear_model.LogisticRegression()


class NaivelyCalibratedXGBoost(XGBClassifier):
    """LinearSVC with `predict_proba` method that naively scales
    `decision_function` output."""

    def fit(self, X, y):
        eval_set = [(xtest, ytest)]
        super().fit(X, y, early_stopping_rounds=10, eval_metric="logloss", eval_set=eval_set, verbose=False)
        df = self.decision_function(X)
        self.df_min_ = df.min()
        self.df_max_ = df.max()

    def predict_proba(self, X):
        """Min-max scale output of `decision_function` to [0,1]."""
        df = self.decision_function(X)
        calibrated_df = (df - self.df_min_) / (self.df_max_ - self.df_min_)
        proba_pos_class = np.clip(calibrated_df, 0, 1)
        proba_neg_class = 1 - proba_pos_class
        proba = np.c_[proba_neg_class, proba_pos_class]
        return proba


xgb = XGBClassifier()
tcf = tree.DecisionTreeClassifier()

clf_list = [
    (lr, "Logistic"),
    (gnb, "Naive Bayes"),
    (svc, "SVC"),
    (rfc, "Random forest"),
    (tcf, "Decision tree"),
    (xgb, "XGBoost")
]

fig = plt.figure(figsize=(10, 10))
gs = GridSpec(6, 2)
colors = plt.cm.get_cmap("Dark2")


ax_calibration_curve = fig.add_subplot(gs[:2, :2])
calibration_displays = {}
for i, (clf, name) in enumerate(clf_list):
    clf.fit(xtrain, ytrain)
    display = CalibrationDisplay.from_estimator(
        clf,
        xtest,
        ytest,
        n_bins=10,
        name=name,
        ax=ax_calibration_curve,
        color=colors(i),
    )
    calibration_displays[name] = display

ax_calibration_curve.grid()
ax_calibration_curve.set_title("Calibration plots")
# Add histogram
grid_positions = [(2, 0), (2, 1), (3, 0), (3, 1), (4, 0), (4, 1)]
for i, (_, name) in enumerate(clf_list):
    row, col = grid_positions[i]
    ax = fig.add_subplot(gs[row, col])

    ax.hist(
        calibration_displays[name].y_prob,
        range=(0, 1),
        bins=10,
        label=name,
        color=colors(i),
    )
    ax.set(title=name, xlabel="Mean predicted probability", ylabel="Count")

plt.tight_layout()
plt.show()
