import os
import csv
import torch
import numpy as np
import random

answer_mapping= {"A": 0, "B": 1, "C": 2, "D": 3}
num_passages = {1: 4, 2: 4, 3: 4, 4: 4, 5: 4, 6: 4, 7: 4, 8: 5, 9: 5, 10: 5, 11: 5, 12: 5, 13: 5, 14: 5, 15: 5}

def read_passages(dir, num_tests=15):
    passage_dictionary = {}
    test_folders = []
    for i in range(num_tests):
        test_folders.append('test' + str(i + 1))
    for test_index, test_folder in enumerate(test_folders):
        passage_files = sorted(filter(lambda folder: not folder.startswith('.'), os.listdir(os.path.join(dir, test_folder))))
        for passage_index, passage_file in enumerate(passage_files):
            with open(os.path.join(dir, test_folder, passage_file)) as file:
                passage_dictionary[(test_index + 1, passage_index + 1)] = file.read()
    return passage_dictionary

def read_questions(dir, num_tests=15):
    question_dictionary = {}
    test_files = []
    for i in range(num_tests):
        test_files.append('test' + str(i + 1) + '.csv')
    for test_index, test_file in enumerate(test_files):
        with open(os.path.join(dir, test_file), newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='"')
            for question_index, row in enumerate(reader):
                if question_index == 0:
                    continue # ignore header
                question, choice_A, choice_B, choice_C, choice_D, answer, passage_index = row
                if (test_index + 1, int(passage_index)) in question_dictionary:
                    question_dictionary[(test_index + 1, int(passage_index))].append([question, choice_A, choice_B, choice_C, choice_D, answer])
                else:
                    question_dictionary[(test_index + 1, int(passage_index))] = [[question, choice_A, choice_B, choice_C, choice_D, answer]]
    return question_dictionary

def get_data(val_numbers, questions):

    raw_input = {
                'questions': [],
                'choice0s':  [],
                'choice1s':  [],
                'choice2s':  [],
                'choice3s':  [],
                'answers':   [],
                'tests':   [],
                'passages':  [],
                }
    
    for test_number in val_numbers:
        for passage_number in range(1, num_passages[test_number] + 1, 1):
            for question in questions[(test_number, passage_number)]:
                raw_input['questions'].append(question[0])
                raw_input['choice0s'].append(question[1])
                raw_input['choice1s'].append(question[2])
                raw_input['choice2s'].append(question[3])
                raw_input['choice3s'].append(question[4])
                raw_input['answers'].append(question[5])
                raw_input['tests'].append(test_number)
                raw_input['passages'].append(passage_number)

    return raw_input

# convert "A" to 0, "B" to 1, "C" to 2, and "D" to 3
def convert_actual_answers(letters):
    numbers = [answer_mapping[i] for i in letters]
    return numbers