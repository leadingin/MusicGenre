# -*- coding:utf-8 -*-

"""
@author: Songgx
@file: 0202_0_create_training_validation_test_set.py
@time: 2016/12/9 6:34
"""

import numpy as np

TOTAL_LINE_NUM = 1000
EACH_GENRE_LINE_NUM = 100
TRAINING_SCALE = 0.6
VALIDATION_SCALE = 0.2

# raw_data.txt 是尾部加了标签的, allRawData.txt 没有标签
# 同样的，scat_data.txt有标签

'''
RAW_FILE_PATH = "merge/raw_data.txt"

TRAINING_FILE_PATH = "tvtsets/training_raw_data.txt"
VALIDATION_FILE_PATH = "tvtsets/validation_raw_data.txt"
TEST_FILE_PATH = "tvtsets/test_raw_data.txt"
'''

RAW_FILE_PATH = "merge/scat_data.txt"

TRAINING_FILE_PATH = "tvtsets/training_scat_data.txt"
VALIDATION_FILE_PATH = "tvtsets/validation_scat_data.txt"
TEST_FILE_PATH = "tvtsets/test_scat_data.txt"

def create_training_validation_test_dataset():
    if TRAINING_SCALE + VALIDATION_SCALE > 1:
        raise ValueError("training_scale + validation_scale > 1")
        exit(1)
    raw_file = open(RAW_FILE_PATH, "r")
    training_file = open(TRAINING_FILE_PATH, "a")
    validation_file = open(VALIDATION_FILE_PATH, "a")
    test_file = open(TEST_FILE_PATH, "a")

    the_genre_line_counter = 0 # 每个风格的当前行数， 到达100时重计数
    index = np.random.permutation(EACH_GENRE_LINE_NUM)
    total_finished = 0
    for i in range(TOTAL_LINE_NUM):
        if the_genre_line_counter > 0 and i % 100 == 0:
            the_genre_line_counter = 0
            index = np.random.permutation(EACH_GENRE_LINE_NUM)

        else:
            the_genre_line_counter += 1

        training_index = index[:int(EACH_GENRE_LINE_NUM * TRAINING_SCALE)]
        validation_index = index[int(EACH_GENRE_LINE_NUM * TRAINING_SCALE):int(EACH_GENRE_LINE_NUM * \
                                                                               (TRAINING_SCALE + VALIDATION_SCALE))]
        test_index = index[int(EACH_GENRE_LINE_NUM * (TRAINING_SCALE + VALIDATION_SCALE)):]

        line_content = raw_file.readline()
        if line_content.strip() == "":
            raise ValueError("Empty Line founded")

        if the_genre_line_counter in training_index:
            training_file.write(line_content)
        elif the_genre_line_counter in validation_index:
            validation_file.write(line_content)
        elif the_genre_line_counter in test_index:
            test_file.write(line_content)

        temp = the_genre_line_counter + 1
        if temp % 100 == 0:
            total_finished += temp
            print("Line {} finished.".format(total_finished))
    raw_file.flush()
    training_file.flush()
    validation_file.flush()
    test_file.flush()

    raw_file.close()
    training_file.close()
    validation_file.close()
    test_file.close()


def verify_dataset(file_path_array):
    for file_path in file_path_array:
        print("Verify file: {} ------------------".format(file_path))
        line_num = count_line_num(file_path)
        print("{} has {} lines".format(file_path, line_num))
        f = open(file_path, "r")
        for i in range(line_num):
            line_num1 = i + 1
            l = f.readline()[-10:]

            if "training" in file_path:
                if (line_num1 + 1) % int(EACH_GENRE_LINE_NUM * TRAINING_SCALE)== 0 or (line_num1 - 1) % \
                        int(EACH_GENRE_LINE_NUM * TRAINING_SCALE) == 0 \
                        or line_num1 % int(EACH_GENRE_LINE_NUM * TRAINING_SCALE) == 0:
                    print("line-" + str(line_num1) + ":" + l.strip())
            elif "validation" in file_path:
                if (line_num1 + 1) % int(EACH_GENRE_LINE_NUM * VALIDATION_SCALE)== 0 or (line_num1 - 1) % \
                        int(EACH_GENRE_LINE_NUM * VALIDATION_SCALE) == 0 \
                        or line_num1 % int(EACH_GENRE_LINE_NUM * VALIDATION_SCALE) == 0:
                    print("line-" + str(line_num1) + ":" + l.strip())
            elif "test" in file_path:
                TEST_SCALE = 1 - TRAINING_SCALE - VALIDATION_SCALE
                if (line_num1 + 1) % int(EACH_GENRE_LINE_NUM * TEST_SCALE)== 0 or (line_num1 - 1) % \
                        int(EACH_GENRE_LINE_NUM * TEST_SCALE) == 0 \
                        or line_num1 % int(EACH_GENRE_LINE_NUM * TEST_SCALE) == 0:
                    print("line-" + str(line_num1) + ":" + l.strip())
        f.close()
        print("Verify file: {} finished ------------------".format(file_path))


def count_line_num(file_path):
    f = open(file_path, "r")
    line_num = 0
    for line in f:
        line_num += 1
    f.close()
    return line_num


if __name__ == "__main__":
    # print(count_line_num("merge/raw_data.txt"))
    create_training_validation_test_dataset()
    verify_dataset([TRAINING_FILE_PATH, VALIDATION_FILE_PATH, TEST_FILE_PATH])
    print ("All Finished")





