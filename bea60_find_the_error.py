import re

import string

import json
import numpy as np
from math import floor


def main():
    # open both files
    data_correct, data_corrupt = [], []
    with open('data//bea60k.repaired.test//correct.txt', encoding="utf-8") as f:
        for i, line in enumerate(f):
            data_correct.append(line)

    with open('data//bea60k.repaired.test//corrupt.txt', encoding="utf-8") as f:
        for i, line in enumerate(f):
            data_corrupt.append(line)

    false_words = []
    truth_words = []

    for i, data in enumerate(zip(data_correct, data_corrupt)):
        x_temp = data[0].split()
        y_temp = data[1].split()
        f_words, t_words = [], []
        for j, splitted_data in enumerate(zip(x_temp, y_temp)):
            if splitted_data[0] != splitted_data[1]:
                t_words.append(splitted_data[0])
                f_words.append(splitted_data[1])

        false_words.append(f_words)
        truth_words.append(t_words)

    # print(false_words)

    truth_dict, false_dict = {}, {}
    for i, data in enumerate(zip(truth_words, false_words)):
        # print(i,data[0],data[1])
        for j, x in enumerate(zip(data[0], data[1])):
            # print(len(truth_dict), len(false_dict), x[0], x[1])
            if x[0].isalpha():
                truth_dict[x[0]] = 0
            if x[1].isalpha():
                false_dict[x[1]] = 1

    x = json.dumps(truth_dict, indent=4)
    json_file = open("data//bea60_words_test_truth.json", "w")
    json_file.write(x)
    json_file.close()

    x = json.dumps(false_dict, indent=4)
    json_file = open("data//bea60_words_test_false.json", "w")
    json_file.write(x)
    json_file.close()

    combo_dict = {**truth_dict, **false_dict}
    x = json.dumps(combo_dict, indent=4)
    json_file = open("data//bea60_words_test_truth_and_false.json", "w")
    json_file.write(x)
    json_file.close()

    pass


punctuation = r"""!"#$%&'()*+,./:;<=>?@[\]^_`{|}~"""

def remove_punctuation(texts):
    '''
    removes the punctuations
    '''
    stripPunct = str.maketrans('', '', punctuation)
    new = [i.translate(stripPunct) for i in texts]
    new = ''.join(new)
    return new


def check_sent_is_alpha(words):

    for word in words :
        if re.match(r'^[A-Za-z0-9*]+$', word):
            #if word.isalpha() == False:
            return True
        else:
            return False

    return True

def main2():
    data_correct, data_corrupt = [], []
    with open('data//bea60k.repaired.val//correct.txt', encoding="utf-8") as f:
        for i, line in enumerate(f):
            data_correct.append(remove_punctuation(line))
            #data_correct.append(line)

    with open('data//bea60k.repaired.val//corrupt.txt', encoding="utf-8") as f:
        for i, line in enumerate(f):
            data_corrupt.append(remove_punctuation(line))
            #data_corrupt.append(line)

    false_words = []
    truth_words = []
    false_texts, truth_texts = [], []

    sentence_len = 5
    for i, data in enumerate(zip(data_correct, data_corrupt)):
        #if i != 29900-1:
        #    continue
        x_temp = data[0].split()
        y_temp = data[1].split()
        f_texts, t_texts = [], []
        if len(x_temp)< sentence_len:
            continue
        for j, splitted_data in enumerate(zip(x_temp, y_temp)):
            if splitted_data[0] != splitted_data[1]:
                if len(x_temp[j - 2: j + 3]) != 5:
                    #print(len(splitted_data[0]))
                    if j == 0:
                        # left side issue
                        print(1)
                        t_texts.append(["*","*", x_temp[j + 1-1], x_temp[j + 2-1], x_temp[j + 3-1]])
                        f_texts.append(["*","*", y_temp[j + 1-1], y_temp[j + 2-1], y_temp[j + 3-1]])
                    elif j == 1:
                        print(2)
                        t_texts.append(["*", x_temp[j-1], x_temp[j + 1-1], x_temp[j + 2-1], x_temp[j + 3-1]])
                        f_texts.append(["*", y_temp[j-1], y_temp[j + 1-1], y_temp[j + 2-1], y_temp[j + 3-1]])

                    elif j == len(x_temp) - 1:
                        print(3)
                        t_texts.append([x_temp[j - 1-1], x_temp[j-1], x_temp[j], "*","*"])
                        f_texts.append([y_temp[j - 1-1], y_temp[j-1], y_temp[j], "*","*"])

                    elif j == len(x_temp) - 2:
                        print(4)
                        t_texts.append([x_temp[j - 1-1], x_temp[j-1], x_temp[j+1-1], x_temp[j+1], "*"])
                        f_texts.append([y_temp[j - 1-1], y_temp[j-1], y_temp[j+1-1],y_temp[j+1],"*" ])
                else:
                    t_texts.append(x_temp[j - 2: j + 3])
                    f_texts.append(y_temp[j - 2: j + 3])

                    #print(f"truth={x_temp[j - 2: j + 3]}, false={y_temp[j - 2: j + 3]}")

        false_texts.append(f_texts)
        truth_texts.append(t_texts)

    # print(false_words)

    truth_dict, false_dict = {}, {}
    for i, data in enumerate(zip(truth_texts, false_texts)):
        # print(i,data[0],data[1])
        for j, x in enumerate(zip(data[0], data[1])):
            print(len(truth_dict), len(false_dict), x[0], x[1])
            print(' '.join(x[0]).isalpha())

            if check_sent_is_alpha(x[0]):
                truth_dict[' '.join(x[0])] = 0

            if check_sent_is_alpha(x[1]):
                false_dict[' '.join(x[1])] = 1

    x = json.dumps(truth_dict, indent=4)
    json_file = open("data//bea60_sentences_test_truth.json", "w")
    json_file.write(x)
    json_file.close()

    x = json.dumps(false_dict, indent=4)
    json_file = open("data//bea60_sentences_test_false.json", "w")
    json_file.write(x)
    json_file.close()

    combo_dict = {**truth_dict, **false_dict}
    x = json.dumps(combo_dict, indent=4)
    json_file = open("data//bea60_sentences_test_truth_and_false.json", "w")
    json_file.write(x)
    json_file.close()


if __name__ == '__main__':
    #main()
    main2()

# extract all these words into a json file
# both correct and incorrect
# with label 0 or 1 for true or false


# there is a mismathc in count of positives and negatives. This is because there are multiple negatives for a single
# positive across diffference sentences.
