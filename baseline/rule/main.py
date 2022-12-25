import os.path

from nltk import SnowballStemmer
import nltk.stem


def get_word(token):
    s = nltk.stem.SnowballStemmer('english')
    return s.stem(token)


def pred_by_rule(cmt_tokens, old_code, new_code):
    cmt_tokens = set([get_word(token) for token in cmt_tokens])
    old_code = set([get_word(token) for token in old_code])
    new_code = set([get_word(token) for token in new_code])
    print(cmt_tokens)
    print(old_code)
    print(new_code)
    before = 0
    after = 0
    for token in cmt_tokens:
        if token in old_code:
            before += 1
        if token in new_code:
            after += 1
    print(before)
    print(after)
    if before == after:
        return 0
    else:
        return 1


def get_tokens(filepath):
    with open(filepath) as f:
        reach_old_comment = False
        reach_old_code = False
        reach_newCode = False
        old_cmt = ''
        old_code = ''
        new_code = ''
        for line in f.readlines():
            if line.startswith('oldComment:'):
                reach_old_comment = True
                continue
            if line.startswith('oldCode:'):
                reach_old_comment = False
                reach_old_code = True
                continue
            if line.startswith('newComment:'):
                reach_old_code = False
                continue
            if line.startswith('newCode:'):
                reach_newCode = True
                continue
            if line.startswith('startline:'):
                reach_newCode = False
                continue
            if line.startswith('label:'):
                label = int(line.split(':')[-1])

            if reach_old_comment:
                old_cmt += line
            if reach_old_code:
                old_code += line
            if reach_newCode:
                new_code += line
        old_cmt = [token for token in (''.join([ch if ch.isalpha() else ' ' for ch in old_cmt])).split(' ') if
                   token != '']
        old_code = [token for token in (''.join([ch if ch.isalpha() else ' ' for ch in old_code])).split(' ') if
                    token != '']
        new_code = [token for token in (''.join([ch if ch.isalpha() else ' ' for ch in new_code])).split(' ') if
                    token != '']
        print(old_cmt)
        print(old_code)
        print(new_code)
        print(label)
        return old_cmt, old_code, new_code, label


TP = 0
FP = 0
TN = 0
FN = 0


def traverse_folder(filepath):
    if os.path.isdir(filepath):
        for f in os.listdir(filepath):
            traverse_folder(os.path.join(filepath, f))
    else:
        if filepath.endswith('.java'):
            old_cmt, old_code, new_code, label = get_tokens(filepath)
            pred = pred_by_rule(old_cmt, old_code, new_code)
            if pred == 1 and label == 1:
                global TP
                TP += 1
            if pred == 1 and label == 0:
                global FP
                FP += 1
            if pred == 0 and label == 1:
                global FN
                FN += 1
            if pred == 0 and label == 0:
                global TN
                TN += 1


traverse_folder("features_path")

print('-------------- result --------------')
print(TP)
print(FP)
print(TN)
print(FN)
pre = TP / (TP + FP)
recall = TP / (TP + FN)
print('pre:', TP / (TP + FP))
print('recall:', TP / (TP + FN))
print('f1', 2 * pre * recall / (pre + recall))
