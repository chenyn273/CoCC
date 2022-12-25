import os

l = set()
cnt = 0
rmcnt = 0


def traverse_folder(url):
    if os.path.isdir(url):
        for f in os.listdir(url):
            traverse_folder(os.path.join(url, f))
    else:
        try:
            global cnt
            cnt += 1

            s = ''
            with open(url) as f:
                reach_old_comment = False
                reach_old_code = False
                reach_new_comment = False
                reach_new_code = False
                old_comment = ''
                old_code = ''
                new_comment = ''
                new_code = ''
                for line in f.readlines():
                    if line.startswith('oldComment:'):
                        reach_old_comment = True
                        continue
                    if line.startswith('oldCode:'):
                        reach_old_code = True
                        reach_old_comment = False
                        continue
                    if line.startswith('newComment:'):
                        reach_new_comment = True
                        reach_old_code = False
                        continue
                    if line.startswith('newCode:'):
                        reach_new_code = True
                        reach_new_comment = False
                        continue
                    if line.startswith('startline:'):
                        break

                    if reach_old_comment:
                        old_comment += line
                    if reach_old_code:
                        old_code += line
                    if reach_new_comment:
                        new_comment += line
                    if reach_new_code:
                        new_code += line

                    new_comment = ''.join([ch if ch.isalpha() else '' for ch in new_comment.replace('\n', '')])
                    new_code = ''.join([ch if ch.isalpha() else '' for ch in new_code.replace('\n', '')])
                    old_comment = ''.join([ch if ch.isalpha() else '' for ch in old_comment.replace('\n', '')])
                    old_code = ''.join([ch if ch.isalpha() else '' for ch in old_code.replace('\n', '')])

                    s += new_code + new_comment + old_code + old_comment
                    # print(s)

            if s in l:
                os.remove(url)
                global rmcnt
                rmcnt += 1
                print('cnt:', cnt)
                print('rmcnt:', rmcnt)
            else:
                l.add(s)
        except:
            pass


if __name__ == '__main__':
    traverse_folder("/Users/chenyn/chenyn's/研究生/DataSet/My/data/changes_drop_duplicates 1")
