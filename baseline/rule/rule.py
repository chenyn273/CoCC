import pandas as pd

print('block-type')
rule = pd.read_csv("./b.csv")

label = rule.label
sim = rule.cmt2cd_sim_change

label = list(label)
sim = list(sim)

pred = list()
for v in sim:
    if v >= 0.05:
        pred.append(1)
    else:
        pred.append(0)

TP = 0
FP = 0
TN = 0
FN = 0

for i in range(0, len(label)):
    if label[i] == 1 and pred[i] == 1:
        TP += 1
    if label[i] == 1 and pred[i] == 0:
        FN += 1
    if label[i] == 0 and pred[i] == 1:
        FP += 1
    if label[i] == 0 and pred[i] == 0:
        TN += 1

pre = TP / (TP + FP)
recall = TP / (TP + FN)
print('pre:', TP / (TP + FP))
print('recall:', TP / (TP + FN))
print('f1', 2 * pre * recall / (pre + recall))

print('method-type')
rule = pd.read_csv("./m.csv")

label = rule.label
sim = rule.cmt2cd_sim_change

label = list(label)
sim = list(sim)

pred = list()
for v in sim:
    if v >= 0.05:
        pred.append(1)
    else:
        pred.append(0)

TP = 0
FP = 0
TN = 0
FN = 0

for i in range(0, len(label)):
    if label[i] == 1 and pred[i] == 1:
        TP += 1
    if label[i] == 1 and pred[i] == 0:
        FN += 1
    if label[i] == 0 and pred[i] == 1:
        FP += 1
    if label[i] == 0 and pred[i] == 0:
        TN += 1

pre = TP / (TP + FP)
recall = TP / (TP + FN)
print('pre:', TP / (TP + FP))
print('recall:', TP / (TP + FN))
print('f1', 2 * pre * recall / (pre + recall))

print('B & M-type')
rule = pd.read_csv("./b&m.csv")

label = rule.label
sim = rule.cmt2cd_sim_change

label = list(label)
sim = list(sim)

pred = list()
for v in sim:
    if v >= 0.05:
        pred.append(1)
    else:
        pred.append(0)

TP = 0
FP = 0
TN = 0
FN = 0

for i in range(0, len(label)):
    if label[i] == 1 and pred[i] == 1:
        TP += 1
    if label[i] == 1 and pred[i] == 0:
        FN += 1
    if label[i] == 0 and pred[i] == 1:
        FP += 1
    if label[i] == 0 and pred[i] == 0:
        TN += 1

pre = TP / (TP + FP)
recall = TP / (TP + FN)
print('pre:', TP / (TP + FP))
print('recall:', TP / (TP + FN))
print('f1', 2 * pre * recall / (pre + recall))
