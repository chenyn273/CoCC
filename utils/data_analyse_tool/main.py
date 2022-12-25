import os


def get_change_info(path):
    is_method = False
    with open(path) as f:
        change_line = []
        for line in f.readlines():
            if line.startswith('type:') and line.__contains__('METHOD_COMMENT'):
                is_method = True
            if line.startswith('changeNum:'):
                change_num = int(line.split(':')[-1])
            if line.startswith('label:'):
                label = int(line.split(':')[-1])
            if line.startswith('change ') and line.__contains__(':') and line.__contains__(','):
                sl, el = [int(x.replace(' ', '').replace('\n', '')) for x in line.split(':')[-1].split(',')]
                while sl <= el:
                    if sl not in change_line:
                        change_line.append(sl)
                    sl += 1
    if is_method:
        return 0, 0, 0
    return change_num, len(change_line), label


change_number = 0
changed_cnt = 0
unchanged_cnt = 0
method_cnt = 0


def traverse_folder_for_Q13(path):
    if os.path.isdir(path):
        for f in os.listdir(path):
            traverse_folder_for_Q13(os.path.join(path, f))
    else:
        if path.endswith('.java'):
            global change_number
            global unchanged_cnt
            global changed_cnt
            global method_cnt
            change_num, st_num, label = get_change_info(path)
            if change_num == 0 and st_num == 0 and label == 0:
                method_cnt += 1
            change_number += change_num
            if label == 1:
                changed_cnt += 1
            else:
                unchanged_cnt += 1

traverse_folder_for_Q13("/Users/chenyn/chenyn's/研究生/DataSet/My dect/data/回复/")
print(method_cnt)

# print(change_number)
# print(changed_cnt / (changed_cnt + unchanged_cnt) * 100)
# print(unchanged_cnt / (changed_cnt + unchanged_cnt) * 100)

cnt = 0
change_num_1_3_label_1 = 0
change_num_1_3_label_0 = 0

change_num_4_6_label_1 = 0
change_num_4_6_label_0 = 0

change_num_7_9_label_1 = 0
change_num_7_9_label_0 = 0

change_num_10_12_label_1 = 0
change_num_10_12_label_0 = 0

change_num_13_15_label_1 = 0
change_num_13_15_label_0 = 0

change_num_over_15_label_1 = 0
change_num_over_15_label_0 = 0

change_st_1_3_label_1 = 0
change_st_1_3_label_0 = 0

change_st_4_6_label_1 = 0
change_st_4_6_label_0 = 0

change_st_7_9_label_1 = 0
change_st_7_9_label_0 = 0

change_st_10_12_label_1 = 0
change_st_10_12_label_0 = 0

change_st_13_15_label_1 = 0
change_st_13_15_label_0 = 0

change_st_over_15_label_1 = 0
change_st_over_15_label_0 = 0


def traverse_folder_for_Q8(path):
    if os.path.isdir(path):
        for f in os.listdir(path):
            traverse_folder_for_Q8(os.path.join(path, f))
    else:
        if path.endswith('.java'):
            global cnt
            cnt += 1
            change_num, st_num, label = get_change_info(path)
            if label == 0:
                if st_num >= 1 and st_num <= 3:
                    global change_st_1_3_label_0
                    change_st_1_3_label_0 += 1
                if st_num >= 4 and st_num <= 6:
                    global change_st_4_6_label_0
                    change_st_4_6_label_0 += 1
                if st_num >= 7 and st_num <= 9:
                    global change_st_7_9_label_0
                    change_st_7_9_label_0 += 1
                if st_num >= 10 and st_num <= 12:
                    global change_st_10_12_label_0
                    change_st_10_12_label_0 += 1
                if st_num >= 13 and st_num <= 15:
                    global change_st_13_15_label_0
                    change_st_13_15_label_0 += 1
                if st_num > 15:
                    global change_st_over_15_label_0
                    change_st_over_15_label_0 += 1

                if change_num >= 1 and change_num <= 3:
                    global change_num_1_3_label_0
                    change_num_1_3_label_0 += 1
                if change_num >= 4 and change_num <= 6:
                    global change_num_4_6_label_0
                    change_num_4_6_label_0 += 1
                if change_num >= 7 and change_num <= 9:
                    global change_num_7_9_label_0
                    change_num_7_9_label_0 += 1
                if change_num >= 10 and change_num <= 12:
                    global change_num_10_12_label_0
                    change_num_10_12_label_0 += 1
                if change_num >= 13 and change_num <= 15:
                    global change_num_13_15_label_0
                    change_num_13_15_label_0 += 1
                if change_num > 15:
                    global change_num_over_15_label_0
                    change_num_over_15_label_0 += 1

            else:
                if st_num >= 1 and st_num <= 3:
                    global change_st_1_3_label_1
                    change_st_1_3_label_1 += 1
                if st_num >= 4 and st_num <= 6:
                    global change_st_4_6_label_1
                    change_st_4_6_label_1 += 1
                if st_num >= 7 and st_num <= 9:
                    global change_st_7_9_label_1
                    change_st_7_9_label_1 += 1
                if st_num >= 10 and st_num <= 12:
                    global change_st_10_12_label_1
                    change_st_10_12_label_1 += 1
                if st_num >= 13 and st_num <= 15:
                    global change_st_13_15_label_1
                    change_st_13_15_label_1 += 1
                if st_num > 15:
                    global change_st_over_15_label_1
                    change_st_over_15_label_1 += 1

                if change_num >= 1 and change_num <= 3:
                    global change_num_1_3_label_1
                    change_num_1_3_label_1 += 1
                if change_num >= 4 and change_num <= 6:
                    global change_num_4_6_label_1
                    change_num_4_6_label_1 += 1
                if change_num >= 7 and change_num <= 9:
                    global change_num_7_9_label_1
                    change_num_7_9_label_1 += 1
                if change_num >= 10 and change_num <= 12:
                    global change_num_10_12_label_1
                    change_num_10_12_label_1 += 1
                if change_num >= 13 and change_num <= 15:
                    global change_num_13_15_label_1
                    change_num_13_15_label_1 += 1
                if change_num > 15:
                    global change_num_over_15_label_1
                    change_num_over_15_label_1 += 1
                pass

# traverse_folder_for_Q8("/Users/chenyn/chenyn's/研究生/DataSet/My dect/data/回复/")
# print('total:', cnt)
# print('---------------------------')
# print(change_num_1_3_label_1)
# print(change_num_1_3_label_0)
# print(change_num_1_3_label_1 / (change_num_1_3_label_1 + change_num_1_3_label_0))
#
# print(change_num_4_6_label_1)
# print(change_num_4_6_label_0)
# print(change_num_4_6_label_1 / (change_num_4_6_label_1 + change_num_4_6_label_0))
#
# print(change_num_7_9_label_1)
# print(change_num_7_9_label_0)
# print(change_num_7_9_label_1 / (change_num_7_9_label_1 + change_num_7_9_label_0))
#
# print(change_num_10_12_label_1)
# print(change_num_10_12_label_0)
# print(change_num_10_12_label_1 / (change_num_10_12_label_1 + change_num_10_12_label_0))
#
# print(change_num_13_15_label_1)
# print(change_num_13_15_label_0)
# print(change_num_13_15_label_1 / (change_num_13_15_label_1 + change_num_13_15_label_0))
#
# print(change_num_over_15_label_1)
# print(change_num_over_15_label_0)
# print(change_num_over_15_label_1 / (change_num_over_15_label_1 + change_num_over_15_label_0))
# print('-----------------------------------')
# print(change_st_1_3_label_1)
# print(change_st_1_3_label_0)
# print(change_st_1_3_label_1 / (change_st_1_3_label_1 + change_st_1_3_label_0))
#
# print(change_st_4_6_label_1)
# print(change_st_4_6_label_0)
# print(change_st_4_6_label_1 / (change_st_4_6_label_1 + change_st_4_6_label_0))
#
# print(change_st_7_9_label_1)
# print(change_st_7_9_label_0)
# print(change_st_7_9_label_1 / (change_st_7_9_label_1 + change_st_7_9_label_0))
#
# print(change_st_10_12_label_1)
# print(change_st_10_12_label_0)
# print(change_st_10_12_label_1 / (change_st_10_12_label_1 + change_st_10_12_label_0))
#
# print(change_st_13_15_label_1)
# print(change_st_13_15_label_0)
# print(change_st_13_15_label_1 / (change_st_13_15_label_1 + change_st_13_15_label_0))
#
# print(change_st_over_15_label_1)
# print(change_st_over_15_label_0)
# print(change_st_over_15_label_1 / (change_st_over_15_label_1 + change_st_over_15_label_0))
