import os
from nltk import word_tokenize
from utils.comment import *
import random
from gensim.models import word2vec, Word2Vec
import numpy as np


# 输入：文件路径
# 输出：从CCSet中返回新旧注释和代码，         return oldComment, oldCode, newComment, newCode
#       如果有空，返回None
def getCommentAndCode(filepath):
    with open(filepath) as f:
        reachOldComment = False
        reachOldCode = False
        reachNewComment = False
        reachNewCode = False
        oldComment = ''
        oldCode = ''
        newComment = ''
        newCode = ''
        for line in f.readlines():
            if line.startswith('oldComment:'):
                reachOldComment = True
                continue
            if line.startswith('oldCode:'):
                reachOldComment = False
                reachOldCode = True
                continue
            if line.startswith('newComment:'):
                reachOldCode = False
                reachNewComment = True
                continue
            if line.startswith('newCode:'):
                reachNewComment = False
                reachNewCode = True
                continue
            if line.startswith('startline:'):
                break
            line = line.strip();
            line = ''.join([ch if ch.isalpha() else ' ' for ch in line])
            if reachOldComment and line.strip() != '':
                oldComment += line + '\n'
            if reachOldCode and line.strip() != '':
                oldCode += line + '\n'
            if reachNewComment and line.strip() != '':
                newComment += line + '\n'
            if reachNewCode and line.strip() != '':
                newCode += line + '\n'
        oldComment = oldComment.replace('\n', '', -1)
        oldCode = oldCode.replace('\n', '', -1)
        newComment = newComment.replace('\n', '', -1)
        newCode = newCode.replace('\n', '', -1)
        if oldComment == '' or oldCode == '' or newCode == '':
            return None
        return oldComment, oldCode, newComment, newCode


# 输入 oldComment,oldCode,newComment,newCode
# 输出 Txt
def getTxt(oldComment, oldCode, newComment, newCode):
    tokenizedOldComment = word_tokenize(oldComment)
    tokenizedOldCode = word_tokenize(oldCode)
    tokenizedNewComment = word_tokenize(newComment)
    tokenizedNewCode = word_tokenize(newCode)
    oldCommentTxt = ''
    oldCodeTxt = ''
    newCommentTxt = ''
    newCodeTxt = ''
    for x in tokenizedOldComment:
        r1 = random.randint(0, len(tokenizedOldCode) - 1)
        r2 = random.randint(0, len(tokenizedOldCode) - 1)
        oldCommentTxt += x + ' ' + tokenizedOldCode[r1] + ' ' + tokenizedOldCode[r2] + ' '

    for x in tokenizedOldCode:
        r1 = random.randint(0, len(tokenizedOldComment) - 1)
        r2 = random.randint(0, len(tokenizedOldComment) - 1)
        oldCodeTxt += x + ' ' + tokenizedOldComment[r1] + ' ' + tokenizedOldComment[r2] + ' '

    for x in tokenizedNewComment:
        r1 = random.randint(0, len(tokenizedNewCode) - 1)
        r2 = random.randint(0, len(tokenizedNewCode) - 1)
        newCommentTxt += x + ' ' + tokenizedNewCode[r1] + ' ' + tokenizedNewCode[r2] + ' '

    for x in tokenizedNewCode:
        r1 = random.randint(0, len(tokenizedNewComment) - 1)
        r2 = random.randint(0, len(tokenizedNewComment) - 1)
        newCodeTxt += x + ' ' + tokenizedNewComment[r1] + ' ' + tokenizedNewComment[r2] + ' '

    return oldCommentTxt + '. ' + oldCodeTxt + '. ' + newCommentTxt + '. ' + newCodeTxt + '. '


# 处理复合词 调用cup
# 输入 txt
# 输出 处理复合词后的语料库txt
def useCupToGetTxt(txt):
    commentCleaner = CommentCleaner(replace_digit=True)
    javaDocDescPreprocessor = JavadocDescPreprocessor(comment_cleaner=commentCleaner)
    result = javaDocDescPreprocessor.preprocess_desc(txt, txt)
    res = ''
    for d in result:
        l = d['src_sent_tokens']
        for token in l:
            if not token.__contains__('<con>'):
                res += token + ' '
    return res


# 输入 ccset java文件路径
# 输出 由此文件生成的语料库
def generateTxt(filepath):
    res = getCommentAndCode(filepath)
    if not res is None:
        oldComment, oldCode, newComment, newCode = res
        txt = getTxt(oldComment, oldCode, newComment, newCode)
        txt = useCupToGetTxt(txt)
        return txt
    else:
        return ''


model = Word2Vec.load("/Users/chenyn/chenyn's/研究生/DataSet/My dect/word2vector/wordvector/word_vector.model")


def sim_word2word(wd1, wd2):
    res = 0.0
    try:
        res = model.wv.similarity(wd1, wd2)
        return res
    except:
        return 0.0


def sim_word2sentence(wd, sentence):
    if len(sentence) > 0:
        return max([sim_word2word(wd, word) for word in sentence])
    # if len(sentence) > 0:
    #     return sum([sim_word2word(wd, word) for word in sentence]) / len(sentence)
    else:
        return 0.0


def sim_sen12sen2(sen1, sen2):
    if len(sen1) == 0 or len(sen2) == 0:
        return 0.0
    l = [sim_word2sentence(wd, sen2) for wd in sen1]
    s = sum(l)
    return s / len(sen1)


def sim_sen2sen(sen1, sen2):
    return (sim_sen12sen2(sen1, sen2) + sim_sen12sen2(sen2, sen1)) / 2


def get_changed_sentence(filepath):
    res = ''
    with open(filepath) as f:
        for line in f.readlines():
            if line.startswith('change_entity_uniqueName:'):
                line = line.replace('change_entity_uniqueName:', '')
                line = line.strip();
                line = ''.join([ch if ch.isalpha() else ' ' for ch in line])
                res += line + '\n'
                res = res.replace('\n', '', -1)

    return useCupToGetTxt(res)


def get_all_token_change_sim(filepath):
    res = getCommentAndCode(filepath)
    try:
        oldComment, oldCode, newComment, newCode = res
    except:
        os.remove(filepath)

    tokenizedOldComment = word_tokenize(oldComment)
    tokenizedOldCode = word_tokenize(oldCode)
    tokenizedNewCode = word_tokenize(newCode)

    oldCommentTxt = ''
    oldCodeTxt = ''
    newCodeTxt = ''

    for x in tokenizedOldComment:
        oldCommentTxt += x + ' '
    for x in tokenizedOldCode:
        oldCodeTxt += x + ' '
    for x in tokenizedNewCode:
        newCodeTxt += x + ' '

    oldComment = useCupToGetTxt(oldCommentTxt)
    oldCode = useCupToGetTxt(oldCodeTxt)
    newCode = useCupToGetTxt(newCodeTxt)

    old2old_sim_t = [sim_word2sentence(word, oldCode.split()) for word in oldComment.split()]
    old2new_sim_t = [sim_word2sentence(word, newCode.split()) for word in oldComment.split()]

    res = []
    for i in range(len(old2old_sim_t)):
        res.append(abs(old2old_sim_t[i] - old2new_sim_t[i]))
    return sum(res)


def get_sims(filepath):
    changed_sen = get_changed_sentence(filepath)
    res = getCommentAndCode(filepath)

    oldComment, oldCode, newComment, newCode = res

    tokenizedOldComment = word_tokenize(oldComment)
    tokenizedOldCode = word_tokenize(oldCode)
    tokenizedNewCode = word_tokenize(newCode)
    tokenizedChangedSen = word_tokenize(changed_sen)

    oldCommentTxt = ''
    oldCodeTxt = ''
    newCodeTxt = ''
    changed_sen = ''

    for x in tokenizedOldComment:
        oldCommentTxt += x + ' '
    for x in tokenizedOldCode:
        oldCodeTxt += x + ' '
    for x in tokenizedNewCode:
        newCodeTxt += x + ' '
    for x in tokenizedChangedSen:
        changed_sen += x + ' '

    oldComment = useCupToGetTxt(oldCommentTxt)
    oldCode = useCupToGetTxt(oldCodeTxt)
    newCode = useCupToGetTxt(newCodeTxt)
    changedsen = useCupToGetTxt(changed_sen)

    return round(sim_sen2sen(oldComment.split(), oldCode.split()), 6), round(
        sim_sen2sen(oldComment.split(), newCode.split()), 6), round(
        sim_sen2sen(oldComment.split(), changedsen.split()), 6)


import os
import re
import nltk
from nltk import word_tokenize
from nltk import pos_tag

remove_chars = '[·’!"\#$%&\'()＃！（）*+,-./:;<=>?\@，：?￥★、…．＞【】［］《》？“”‘’\[\\]^_`{|}~]+'


def getTokens(sentence):
    sentence = re.sub(remove_chars, "", sentence)
    sentence = ''.join([i for i in sentence if not i.isdigit()])
    words = word_tokenize(sentence)
    return words


import os


def getFeatures(filepath):
    with open(filepath) as f:
        all_token_change_sim = get_all_token_change_sim(filepath)
        cmt2cd_sim_before, cmt2cd_sim_after, cmt2ch_sim = get_sims(filepath)
        sim_change = abs(cmt2cd_sim_before - cmt2cd_sim_after)
        label = 0
        changeNum = 0  ##变更数量
        attribute = 0  ##
        methodDeclaration = 0  ##
        methodRenaming = 0  ##
        returnType = 0  ##
        parameterDelete = 0  ##
        parameterInsert = 0  ##
        parameterRenaming = 0  ##
        parameterTypeChange = 0  ##
        containReturn = 0  ##
        lineNumOfOldCode = 0  ##
        lineNumOfOldComment = 0  ##
        lineNumOfNewCode = 0  ##
        oldCode = ''
        reachOldCode = False
        oldComment = ''
        reachOldComment = False
        newCode = ''
        reachNewCode = False
        TODOCount = 0  ##
        XXXCount = 0  ##
        BUGCount = 0  ##
        VERSIONCount = 0  ##
        FIXMECount = 0  ##
        FIXEDCount = 0  ##
        commentByCCSet = 0  ##
        lineNumOfChanged = 0  ##
        setOfChangedLine = set()
        changedLineByAllCodeLine = 0.0  ##
        changeCase = ''

        ifInsert = 0  ##
        ifUpdate = 0  ##
        ifMove = 0  ##
        ifDelete = 0  ##

        forInsert = 0  ##
        forUpdate = 0  ##
        forMove = 0  ##
        forDelete = 0  ##

        foreachInsert = 0  ##
        foreachUpdate = 0  ##
        foreachMove = 0  ##
        foreachDelete = 0  ##

        whileInsert = 0  ##
        whileUpdate = 0  ##
        whileMove = 0  ##
        whileDelete = 0  ##

        catchInsert = 0  ##
        catchUpdate = 0  ##
        catchMove = 0  ##
        catchDelete = 0  ##

        tryInsert = 0  ##
        tryUpdate = 0  ##
        tryMove = 0  ##
        tryDelete = 0  ##

        throwInsert = 0  ##
        throwUpdate = 0  ##
        throwMove = 0  ##
        throwDelete = 0  ##

        methodInvInsert = 0  ##
        methodInvUpdate = 0  ##
        methodInvMove = 0  ##
        methodInvDelete = 0  ##

        assignInsert = 0  ##
        assignUpdate = 0  ##
        assignMove = 0  ##
        assignDelete = 0  ##

        varDecInsert = 0  ##
        varDecUpdate = 0  ##
        varDecMove = 0  ##
        varDecDelete = 0  ##

        elseInsert = 0  ##
        elseUpdate = 0  ##
        elseMove = 0  ##
        elseDelete = 0  ##

        oldCommentPos = dict()  ##
        codePosChange = dict()  ##
        bothHavePairNumChange = 0  ##

        origin = f.readlines()
        for line in origin:
            if re.match(r'change \d : \d+,\d+', line):
                matched = re.match(r'change \d : (\d+),(\d+)', line)
                for i in range(int(matched.group(1)), 1 + int(matched.group(2))):
                    setOfChangedLine.add(i)
            if re.match(r'label:\d', line):
                label = str(re.match(r'label:(\d)', line).group(1))
            if line.startswith('oldComment:'):
                reachOldComment = True
                continue
            if line.startswith('oldCode:'):
                reachOldCode = True
                reachOldComment = False
                continue
            if line.startswith('newCode:'):
                reachNewCode = True
                continue
            if line.startswith('startline:'):
                reachNewCode = False
                continue
            if line.startswith('newComment:'):
                reachOldCode = False
            if reachOldCode:
                if len(line.strip()) != 0:
                    oldCode += line
                    lineNumOfOldCode += 1
            if reachNewCode:
                if len(line.strip()):
                    newCode += line
                    lineNumOfNewCode += 1

            if reachOldComment:
                if len(line.strip()) != 0:
                    oldComment += line
                    lineNumOfOldComment += 1
            if line.startswith('change_entity_type:'):
                if not (line.__contains__('COMMENT') or line.__contains__('DOC')):
                    changeNum += 1
            if line.startswith('change_type:') and line.__contains__('ATTRIBUTE'):
                attribute += 1
            if line.startswith('change_entity:') and line.__contains__('METHOD_DECLARATION'):
                methodDeclaration += 1
            if line.startswith('change_type:'):
                if line.__contains__('METHOD_RENAMING'):
                    methodRenaming += 1
                if line.__contains__('RETURN_TYPE'):
                    returnType += 1
                if line.__contains__('PARAMETER_DELETE'):
                    parameterDelete += 1
                if line.__contains__('PARAMETER_INSERT'):
                    parameterInsert += 1
                if line.__contains__('PARAMETER_RENAMING'):
                    parameterRenaming += 1
                if line.__contains__('PARAMETER_TYPE_CHANGE'):
                    parameterTypeChange += 1
            if re.match(r'change:[A-z]+:', line):
                matched = re.match(r'change:([A-z]+):', line).group(1)
                changeCase = str(matched)

            if line.startswith('change_entity_type:') and line.__contains__('IF_STATEMENT') and changeCase.__contains__(
                    'Update'):
                ifUpdate += 1
            if line.startswith('change_entity_type:') and line.__contains__('IF_STATEMENT') and changeCase.__contains__(
                    'Move'):
                ifMove += 1
            if line.startswith('change_entity_type:') and line.__contains__('IF_STATEMENT') and changeCase.__contains__(
                    'Delete'):
                ifDelete += 1
            if line.startswith('change_entity_type:') and line.__contains__('IF_STATEMENT') and changeCase.__contains__(
                    'Insert'):
                ifInsert += 1

            if line.startswith('change_entity_type:') and line.__contains__(
                    'FOR_STATEMENT') and changeCase.__contains__('Update'):
                forUpdate += 1
            if line.startswith('change_entity_type:') and line.__contains__(
                    'FOR_STATEMENT') and changeCase.__contains__('Move'):
                forMove += 1
            if line.startswith('change_entity_type:') and line.__contains__(
                    'FOR_STATEMENT') and changeCase.__contains__('Delete'):
                forDelete += 1
            if line.startswith('change_entity_type:') and line.__contains__(
                    'FOR_STATEMENT') and changeCase.__contains__('Insert'):
                forInsert += 1

            if line.startswith('change_entity_type') and line.__contains__(
                    'FOREACH_STATEMENT') and changeCase.__contains__('Update'):
                foreachUpdate += 1
            if line.startswith('change_entity_type') and line.__contains__(
                    'FOREACH_STATEMENT') and changeCase.__contains__('Move'):
                foreachMove += 1
            if line.startswith('change_entity_type') and line.__contains__(
                    'FOREACH_STATEMENT') and changeCase.__contains__('Delete'):
                foreachDelete += 1
            if line.startswith('change_entity_type') and line.__contains__(
                    'FOREACH_STATEMENT') and changeCase.__contains__('Insert'):
                foreachInsert += 1

            if line.startswith('change_entity_type') and line.__contains__(
                    'WHILE_STATEMENT') and changeCase.__contains__('Update'):
                whileUpdate += 1
            if line.startswith('change_entity_type') and line.__contains__(
                    'WHILE_STATEMENT') and changeCase.__contains__('Move'):
                whileMove += 1
            if line.startswith('change_entity_type') and line.__contains__(
                    'WHILE_STATEMENT') and changeCase.__contains__('Delete'):
                whileDelete += 1
            if line.startswith('change_entity_type') and line.__contains__(
                    'WHILE_STATEMENT') and changeCase.__contains__('Insert'):
                whileInsert += 1

            if line.startswith('change_entity_type') and line.__contains__('CATCH_CLAUSE') and changeCase.__contains__(
                    'Update'):
                catchUpdate += 1
            if line.startswith('change_entity_type') and line.__contains__('CATCH_CLAUSE') and changeCase.__contains__(
                    'Move'):
                catchMove += 1
            if line.startswith('change_entity_type') and line.__contains__('CATCH_CLAUSE') and changeCase.__contains__(
                    'Delete'):
                catchDelete += 1
            if line.startswith('change_entity_type') and line.__contains__('CATCH_CLAUSE') and changeCase.__contains__(
                    'Insert'):
                catchInsert += 1

            if line.startswith('change_entity_type') and line.__contains__(
                    'TRY_STATEMENT') and changeCase.__contains__('Update'):
                tryUpdate += 1
            if line.startswith('change_entity_type') and line.__contains__(
                    'TRY_STATEMENT') and changeCase.__contains__('Move'):
                tryMove += 1
            if line.startswith('change_entity_type') and line.__contains__(
                    'TRY_STATEMENT') and changeCase.__contains__('Delete'):
                tryDelete += 1
            if line.startswith('change_entity_type') and line.__contains__(
                    'TRY_STATEMENT') and changeCase.__contains__('Insert'):
                tryInsert += 1

            if line.startswith('change_entity_type') and line.__contains__(
                    'THROW_STATEMENT') and changeCase.__contains__('Update'):
                throwUpdate += 1
            if line.startswith('change_entity_type') and line.__contains__(
                    'THROW_STATEMENT') and changeCase.__contains__('Move'):
                throwMove += 1
            if line.startswith('change_entity_type') and line.__contains__(
                    'THROW_STATEMENT') and changeCase.__contains__('Delete'):
                throwDelete += 1
            if line.startswith('change_entity_type') and line.__contains__(
                    'THROW_STATEMENT') and changeCase.__contains__('Insert'):
                throwInsert += 1

            if line.startswith('change_entity_type') and line.__contains__(
                    'METHOD_INVOCATION') and changeCase.__contains__('Update'):
                methodInvUpdate += 1
            if line.startswith('change_entity_type') and line.__contains__(
                    'METHOD_INVOCATION') and changeCase.__contains__('Move'):
                methodInvMove += 1
            if line.startswith('change_entity_type') and line.__contains__(
                    'METHOD_INVOCATION') and changeCase.__contains__('Delete'):
                methodInvDelete += 1
            if line.startswith('change_entity_type') and line.__contains__(
                    'METHOD_INVOCATION') and changeCase.__contains__('Insert'):
                methodInvInsert += 1

            if line.startswith('change_entity_type') and line.__contains__('ASSIGNMENT') and changeCase.__contains__(
                    'Update'):
                assignUpdate += 1
            if line.startswith('change_entity_type') and line.__contains__('ASSIGNMENT') and changeCase.__contains__(
                    'Move'):
                assignMove += 1
            if line.startswith('change_entity_type') and line.__contains__('ASSIGNMENT') and changeCase.__contains__(
                    'Delete'):
                assignDelete += 1
            if line.startswith('change_entity_type') and line.__contains__('ASSIGNMENT') and changeCase.__contains__(
                    'Insert'):
                assignInsert += 1

            if line.startswith('change_entity_type') and line.__contains__(
                    'VARIABLE_DECLARATION_STATEMENT') and changeCase.__contains__('Update'):
                varDecUpdate += 1
            if line.startswith('change_entity_type') and line.__contains__(
                    'VARIABLE_DECLARATION_STATEMENT') and changeCase.__contains__('Move'):
                varDecMove += 1
            if line.startswith('change_entity_type') and line.__contains__(
                    'VARIABLE_DECLARATION_STATEMENT') and changeCase.__contains__('Delete'):
                varDecDelete += 1
            if line.startswith('change_entity_type') and line.__contains__(
                    'VARIABLE_DECLARATION_STATEMENT') and changeCase.__contains__('Insert'):
                varDecInsert += 1

            if line.startswith('change_entity_type') and line.__contains__(
                    'ELSE_STATEMENT') and changeCase.__contains__('Update'):
                elseUpdate += 1
            if line.startswith('change_entity_type') and line.__contains__(
                    'ELSE_STATEMENT') and changeCase.__contains__('Move'):
                elseMove += 1
            if line.startswith('change_entity_type') and line.__contains__(
                    'ELSE_STATEMENT') and changeCase.__contains__('Delete'):
                elseDelete += 1
            if line.startswith('change_entity_type') and line.__contains__(
                    'ELSE_STATEMENT') and changeCase.__contains__('Insert'):
                elseInsert += 1

        if oldCode.__contains__('return'):
            containReturn = 1

        TODOCount = 0 if oldComment.upper().count('TODO') == 0 else 1
        FIXMECount = 0 if oldComment.upper().count('FIXME') == 0 else 1
        XXXCount = 0 if oldComment.upper().count('XXX') == 0 else 1
        BUGCount = 0 if oldComment.upper().count('BUG') == 0 else 1
        VERSIONCount = 0 if oldComment.upper().count('VERSION') == 0 else 1
        FIXEDCount = 0 if oldComment.upper().count('FIXED') == 0 else 1
        commentByCCSet = float(format(lineNumOfOldComment / (lineNumOfOldCode + lineNumOfOldComment), '.6f'))
        lineNumOfChanged = len(setOfChangedLine)
        if lineNumOfOldCode != 0:
            changedLineByAllCodeLine = float(format((lineNumOfChanged / lineNumOfOldCode), '.6f'))
        else:
            os.remove(filepath)

        dictOfOldCommentPos = {'NN': 0, 'VB': 0, 'DT': 0, 'IN': 0, 'JJ': 0, 'RB': 0, 'PRP': 0, 'MD': 0, 'LS': 0,
                               'RP': 0}
        dictOfOldCodePos = {'NN': 0, 'VB': 0, 'DT': 0, 'IN': 0, 'JJ': 0, 'RB': 0, 'PRP': 0, 'MD': 0, 'LS': 0, 'RP': 0}
        dictOfNewCodePos = {'NN': 0, 'VB': 0, 'DT': 0, 'IN': 0, 'JJ': 0, 'RB': 0, 'PRP': 0, 'MD': 0, 'LS': 0, 'RP': 0}

        oldCommentPos = pos_tag(getTokens(oldComment))
        oldCodePos = pos_tag(getTokens(oldCode))
        newCodePos = pos_tag(getTokens(newCode))

        bothHaveBefore = [w for w in getTokens(oldComment) if w in getTokens(oldCode)]
        bothHaveAfter = [w for w in getTokens(oldComment) if w in getTokens(newCode)]
        w1 = [w for w in bothHaveBefore if w not in bothHaveAfter]
        w2 = [w for w in bothHaveAfter if w not in bothHaveBefore]
        bothHavePairNumChange = abs(len(w1) + len(w2))

        for x in oldCommentPos:
            if (str(x[1]) == 'NN' or
                    str(x[1]) == 'NNS' or
                    str(x[1]) == 'NNP' or
                    str(x[1]) == 'NNPS'):
                dictOfOldCommentPos['NN'] += 1
            if (str(x[1]) == 'VB' or
                    str(x[1]) == 'VBD' or
                    str(x[1]) == 'VBG' or
                    str(x[1]) == 'VBN' or
                    str(x[1]) == 'VBP' or
                    str(x[1]) == 'VBZ'):
                dictOfOldCommentPos['VB'] += 1
            if (str(x[1]) == 'DT' or
                    str(x[1]) == 'WDT'):
                dictOfOldCommentPos['DT'] += 1
            if (str(x[1]) == 'IN' or
                    str(x[1]) == 'CC'):
                dictOfOldCommentPos['IN'] += 1
            if (str(x[1]) == 'JJ' or
                    str(x[1]) == 'JJR' or
                    str(x[1]) == 'JJS'):
                dictOfOldCommentPos['JJ'] += 1
            if (str(x[1]) == 'RB' or
                    str(x[1]) == 'RBR' or
                    str(x[1]) == 'RBS' or
                    str(x[1]) == 'WRB'):
                dictOfOldCommentPos['RB'] += 1
            if (str(x[1]) == 'PRP' or
                    str(x[1]) == 'PRP$' or
                    str(x[1]) == 'WP' or
                    str(x[1]) == 'WP$'):
                dictOfOldCommentPos['PRP'] += 1
            if (str(x[1]) == 'MD'):
                dictOfOldCommentPos['MD'] += 1
            if (str(x[1]) == 'LS'):
                dictOfOldCommentPos['LS'] += 1
            if (str(x[1]) == 'RP'):
                dictOfOldCommentPos['RP'] += 1

        for x in oldCodePos:
            if (str(x[1]) == 'NN' or
                    str(x[1]) == 'NNS' or
                    str(x[1]) == 'NNP' or
                    str(x[1]) == 'NNPS'):
                dictOfOldCodePos['NN'] += 1
            if (str(x[1]) == 'VB' or
                    str(x[1]) == 'VBD' or
                    str(x[1]) == 'VBG' or
                    str(x[1]) == 'VBN' or
                    str(x[1]) == 'VBP' or
                    str(x[1]) == 'VBZ'):
                dictOfOldCodePos['VB'] += 1
            if (str(x[1]) == 'DT' or
                    str(x[1]) == 'WDT'):
                dictOfOldCodePos['DT'] += 1
            if (str(x[1]) == 'IN' or
                    str(x[1]) == 'CC'):
                dictOfOldCodePos['IN'] += 1
            if (str(x[1]) == 'JJ' or
                    str(x[1]) == 'JJR' or
                    str(x[1]) == 'JJS'):
                dictOfOldCodePos['JJ'] += 1
            if (str(x[1]) == 'RB' or
                    str(x[1]) == 'RBR' or
                    str(x[1]) == 'RBS' or
                    str(x[1]) == 'WRB'):
                dictOfOldCodePos['RB'] += 1
            if (str(x[1]) == 'PRP' or
                    str(x[1]) == 'PRP$' or
                    str(x[1]) == 'WP' or
                    str(x[1]) == 'WP$'):
                dictOfOldCodePos['PRP'] += 1
            if (str(x[1]) == 'MD'):
                dictOfOldCodePos['MD'] += 1
            if (str(x[1]) == 'LS'):
                dictOfOldCodePos['LS'] += 1
            if (str(x[1]) == 'RP'):
                dictOfOldCodePos['RP'] += 1
        for x in newCodePos:
            if (str(x[1]) == 'NN' or
                    str(x[1]) == 'NNS' or
                    str(x[1]) == 'NNP' or
                    str(x[1]) == 'NNPS'):
                dictOfNewCodePos['NN'] += 1
            if (str(x[1]) == 'VB' or
                    str(x[1]) == 'VBD' or
                    str(x[1]) == 'VBG' or
                    str(x[1]) == 'VBN' or
                    str(x[1]) == 'VBP' or
                    str(x[1]) == 'VBZ'):
                dictOfNewCodePos['VB'] += 1
            if (str(x[1]) == 'DT' or
                    str(x[1]) == 'WDT'):
                dictOfNewCodePos['DT'] += 1
            if (str(x[1]) == 'IN' or
                    str(x[1]) == 'CC'):
                dictOfNewCodePos['IN'] += 1
            if (str(x[1]) == 'JJ' or
                    str(x[1]) == 'JJR' or
                    str(x[1]) == 'JJS'):
                dictOfNewCodePos['JJ'] += 1
            if (str(x[1]) == 'RB' or
                    str(x[1]) == 'RBR' or
                    str(x[1]) == 'RBS' or
                    str(x[1]) == 'WRB'):
                dictOfNewCodePos['RB'] += 1
            if (str(x[1]) == 'PRP' or
                    str(x[1]) == 'PRP$' or
                    str(x[1]) == 'WP' or
                    str(x[1]) == 'WP$'):
                dictOfNewCodePos['PRP'] += 1
            if (str(x[1]) == 'MD'):
                dictOfNewCodePos['MD'] += 1
            if (str(x[1]) == 'LS'):
                dictOfNewCodePos['LS'] += 1
            if (str(x[1]) == 'RP'):
                dictOfNewCodePos['RP'] += 1

        for key in dictOfOldCommentPos.keys():
            if len(getTokens(oldComment)) != 0:
                dictOfOldCommentPos[key] = format(dictOfOldCommentPos[key] / len(getTokens(oldComment)), '.6f')
            else:
                os.remove(filepath)
        for key in dictOfOldCodePos.keys():
            if len(getTokens(oldCode)) != 0:
                dictOfOldCodePos[key] = format(dictOfOldCodePos[key] / len(getTokens(oldCode)), '.6f')
            else:
                os.remove(filepath)
        for key in dictOfNewCodePos.keys():
            if len(getTokens(newCode)) != 0:
                dictOfNewCodePos[key] = format(dictOfNewCodePos[key] / len(getTokens(newCode)), '.6f')
            else:
                os.remove(filepath)
        for key1 in dictOfNewCodePos.keys():
            codePosChange[key1] = format(abs(float(dictOfNewCodePos[key1]) - float(dictOfOldCodePos[key1])), '.6f')

        oldCommentPos = dictOfOldCommentPos

        # newfile = open(
        #     ("/Users/chenyn/chenyn's/研究生/DataSet/My dect/features/block/") +
        #     os.path.split(filepath)[
        #         1],
        #     'w')
        newfile = open(
            ("/Users/chenyn/chenyn's/研究生/DataSet/My dect/RQ4/opennms_feature/") +
            os.path.split(filepath)[
                1],
            'w')
        for line in origin:
            newfile.write(line)
        newfile.write('\n-------------------------------\n')
        newfile.write('label:' + str(label) + '\n')
        newfile.write('changeNum:' + str(changeNum) + '\n')  # ok
        newfile.write('attribute:' + str(attribute) + '\n')
        newfile.write('methodDeclaration:' + str(methodDeclaration) + '\n')
        newfile.write('methodRenaming:' + str(methodRenaming) + '\n')
        newfile.write('returnType:' + str(returnType) + '\n')
        newfile.write('parameterDelete:' + str(parameterDelete) + '\n')
        newfile.write('parameterInsert:' + str(parameterInsert) + '\n')
        newfile.write('parameterRenaming:' + str(parameterRenaming) + '\n')
        newfile.write('parameterTypeChange:' + str(parameterTypeChange) + '\n')
        newfile.write('containReturn:' + str(containReturn) + '\n')
        newfile.write('lineNumOfOldCodeBylineNumOfOldCCSet:' + str(
            lineNumOfOldCode / (lineNumOfOldCode + lineNumOfOldComment)) + '\n')
        newfile.write('lineNumOfOldCode:' + str(lineNumOfOldCode) + '\n')
        newfile.write('lineNumOfOldCommentBylineNumOfOldCCSet:' + str(
            lineNumOfOldComment / (lineNumOfOldCode + lineNumOfOldComment)) + '\n')
        newfile.write('lineNumOfOldComment:' + str(lineNumOfOldComment) + '\n')
        newfile.write('TODOCount:' + str(TODOCount) + '\n')
        newfile.write('FIXMECount:' + str(FIXMECount) + '\n')
        newfile.write('XXXCount:' + str(XXXCount) + '\n')
        newfile.write('BUGCount:' + str(BUGCount) + '\n')
        newfile.write('VERSIONCount:' + str(VERSIONCount) + '\n')
        newfile.write('FIXEDCount:' + str(FIXEDCount) + '\n')  # ok
        newfile.write('lineNumOfChanged:' + str(lineNumOfChanged) + '\n')  # ok
        newfile.write('changedLineByAllCodeLine:' + str(changedLineByAllCodeLine) + '\n')  # ok
        newfile.write('ifInsert:' + str(ifInsert) + '\n')
        newfile.write('ifUpdate:' + str(ifUpdate) + '\n')
        newfile.write('ifMove:' + str(ifMove) + '\n')
        newfile.write('ifDelete:' + str(ifDelete) + '\n')  #

        newfile.write('forInsert:' + str(forInsert) + '\n')  #
        newfile.write('forUpdate:' + str(forUpdate) + '\n')  # forUpdate = 0  ##
        newfile.write('forMove:' + str(forMove) + '\n')  # forMove = 0  ##
        newfile.write('forDelete:' + str(forDelete) + '\n')  # forDelete = 0  ##

        newfile.write('foreachInsert:' + str(foreachInsert) + '\n')  # foreachInsert = 0  ##
        newfile.write('foreachUpdate:' + str(foreachUpdate) + '\n')  # foreachUpdate = 0  ##
        newfile.write('foreachMove:' + str(foreachMove) + '\n')  # foreachMove = 0  ##
        newfile.write('foreachDelete:' + str(foreachDelete) + '\n')  # foreachDelete = 0  ##

        newfile.write('whileInsert:' + str(whileInsert) + '\n')  # whileInsert = 0  ##
        newfile.write('whileUpdate:' + str(whileUpdate) + '\n')  # whileUpdate = 0  ##
        newfile.write('whileMove:' + str(whileMove) + '\n')  # whileMove = 0  ##
        newfile.write('whileDelete:' + str(whileDelete) + '\n')  # whileDelete = 0  ##

        newfile.write('catchInsert:' + str(catchInsert) + '\n')  # catchInsert = 0  ##

        newfile.write('catchUpdate:' + str(catchUpdate) + '\n')  # catchUpdate = 0  ##
        newfile.write('catchMove:' + str(catchMove) + '\n')  # catchMove = 0  ##
        newfile.write('catchDelete:' + str(catchDelete) + '\n')  # catchDelete = 0  ##

        newfile.write('tryInsert:' + str(tryInsert) + '\n')  # tryInsert = 0  ##
        newfile.write('tryUpdate:' + str(tryUpdate) + '\n')  # tryUpdate = 0  ##
        newfile.write('tryMove:' + str(tryMove) + '\n')  # tryMove = 0  ##
        newfile.write('tryDelete:' + str(tryDelete) + '\n')  # tryDelete = 0  ##

        newfile.write('throwInsert:' + str(throwInsert) + '\n')  # throwInsert = 0  ##
        newfile.write('throwUpdate:' + str(throwUpdate) + '\n')  # throwUpdate = 0  ##
        newfile.write('throwMove:' + str(throwMove) + '\n')  # throwMove = 0  ##
        newfile.write('throwDelete:' + str(throwDelete) + '\n')  # throwDelete = 0  ##

        newfile.write('methodInvInsert:' + str(methodInvInsert) + '\n')  # methodInvInsert = 0  ##
        newfile.write('methodInvUpdate:' + str(methodInvUpdate) + '\n')  # methodInvUpdate = 0  ##
        newfile.write('methodInvMove:' + str(methodInvMove) + '\n')  # methodInvMove = 0  ##
        newfile.write('methodInvDelete:' + str(methodInvDelete) + '\n')  # methodInvDelete = 0  ##

        newfile.write('assignInsert:' + str(assignInsert) + '\n')  # assignInsert = 0  ##
        newfile.write('assignUpdate:' + str(assignUpdate) + '\n')  # assignUpdate = 0  ##
        newfile.write('assignMove:' + str(assignMove) + '\n')  # assignMove = 0  ##
        newfile.write('assignDelete:' + str(assignDelete) + '\n')  # assignDelete = 0  ##

        newfile.write('varDecInsert:' + str(varDecInsert) + '\n')  # varDecInsert = 0  ##
        newfile.write('varDecUpdate:' + str(varDecUpdate) + '\n')  # varDecUpdate = 0z  ##
        newfile.write('varDecMove:' + str(varDecMove) + '\n')  # varDecMove = 0  ##
        newfile.write('varDecDelete:' + str(varDecDelete) + '\n')  # varDecDelete = 0  ##

        newfile.write('elseInsert:' + str(elseInsert) + '\n')  # elseInsert = 0  ##
        newfile.write('elseUpdate:' + str(elseUpdate) + '\n')  # elseUpdate = 0  ##
        newfile.write('elseMove:' + str(elseMove) + '\n')  # elseMove = 0  ##
        newfile.write('elseDelete:' + str(elseDelete) + '\n')  # elseDelete = 0  ##
        for key in oldCommentPos.keys():
            newfile.write(str(key) + 'Comment:' + str(oldCommentPos[key]) + '\n')
        for key in codePosChange.keys():
            newfile.write(str(key) + 'Code:' + str(codePosChange[key]) + '\n')  # ok
        newfile.write('bothHavePairNumChange:' + str(bothHavePairNumChange) + '\n')  # ok
        newfile.write('cmt2cd_sim_before:' + str(round(abs(cmt2cd_sim_before), 6)) + '\n')
        newfile.write('cmt2cd_sim_after:' + str(round(abs(cmt2cd_sim_after), 6)) + '\n')
        newfile.write('cmt2cd_sim_change:' + str(round(abs(cmt2cd_sim_before - cmt2cd_sim_after), 6)) + '\n')  # ok
        newfile.write('cmt2ch_sim_change:' + str(cmt2ch_sim) + '\n')  # ok
        newfile.write('all_token_change_sim:' + str(round(all_token_change_sim, 6)) + '\n')  # ok

        newfile.close()


count = 0


def blwjj(filepath):
    if os.path.isdir(filepath):
        for f in os.listdir(filepath):
            blwjj(os.path.join(filepath, f))
    else:
        if filepath.endswith('.java'):
            global count
            # print(count)
            count += 1
            getFeatures(os.path.abspath(filepath))


# blwjj("/Users/chenyn/chenyn's/研究生/DataSet/My dect/data/BLOCK")
blwjj("/Users/chenyn/chenyn's/研究生/DataSet/My dect/RQ4/opennms_change")
