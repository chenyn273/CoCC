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


def random_forest_pred():
    rfc = XGBClassifier()
    rfc = rfc.fit(xtrain, ytrain)

    predicted = rfc.predict(xtest)
    print('Recall:', recall_score(ytest, predicted))
    print('F1-score:', f1_score(ytest, predicted))
    print('Precision score:', precision_score(ytest, predicted))


# different_model()
random_forest_pred()
