import csv
import os
from os import path

title = ['label',
         'changeNum',
         'attribute',
         'methodDeclaration',
         'methodRenaming',
         'returnType',
         'parameterDelete',
         'parameterInsert',
         'parameterRenaming',
         'parameterTypeChange',
         'containReturn',
         'lineNumOfOldCodeBylineNumOfOldCCSet',
         'lineNumOfOldCode',
         'lineNumOfOldCommentBylineNumOf"OldCCSet',
         'lineNumOfOldComment',
         'TODOCount',
         'FIXMECount',
         'XXXCount',
         'BUGCount',
         'VERSIONCount',
         'FIXEDCount',
         'lineNumOfChanged',
         'changedLineByAllCodeLine',
         'ifInsert',
         'ifUpdate',
         'ifMove',
         'ifDelete',
         'forInsert',
         'forUpdate',
         'forMove',
         'forDelete',
         'foreachInsert',
         'foreachUpdate',
         'foreachMove',
         'foreachDelete',
         'whileInsert',
         'whileUpdate',
         'whileMove',
         'whileDelete',
         'catchInsert',
         'catchUpdate',
         'catchMove',
         'catchDelete',
         'tryInsert',
         'tryUpdate',
         'tryMove',
         'tryDelete',
         'throwInsert',
         'throwUpdate',
         'throwMove',
         'throwDelete',
         'methodInvInsert',
         'methodInvUpdate',
         'methodInvMove',
         'methodInvDelete',
         'assignInsert',
         'assignUpdate',
         'assignMove',
         'assignDelete',
         'varDecInsert',
         'varDecUpdate',
         'varDecMove',
         'varDecDelete',
         'elseInsert',
         'elseUpdate',
         'elseMove',
         'elseDelete',
         'NNComment',
         'VBComment',
         'DTComment',
         'INComment',
         'JJComment',
         'RBComment',
         'PRPComment',
         'MDComment',
         'LSComment',
         'RPComment',
         'NNCode',
         'VBCode',
         'DTCode',
         'INCode',
         'JJCode',
         'RBCode',
         'PRPCode',
         'MDCode',
         'LSCode',
         'RPCode',
         'bothHavePairNumChange',
         'cmt2cd_sim_before',
         'cmt2cd_sim_after',
         'cmt2cd_sim_change',
         'cmt2ch_sim_change',
         'all_token_change_sim']

with open("/Users/chenyn/chenyn's/研究生/DataSet/My dect/RQ4/ejbca.csv", 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(title)

    url = "/Users/chenyn/chenyn's/研究生/DataSet/My dect/RQ4/ejbca_feature"
    file = os.listdir(url)

    for f in file:
        real_url = path.join(url, f)
        if path.isfile(real_url):
            pathname = str(path.abspath(real_url))
            if (pathname.endswith('.java')):
                with open(pathname, 'r') as f:
                    row = []
                    start = False
                    for line in f:
                        if line.startswith('-------------------------------'):
                            start = True
                        if line.__contains__(':') and start:
                            print(line.split(':')[1])
                            row.append(str(line.split(':')[1]).replace('\n', ''))
                        if line.startswith('all_token_change_sim:'):
                            # row.append(str(''.join(pathname.split('/')[-3:])))
                            break
                    print(row)
                    writer.writerow(row)
