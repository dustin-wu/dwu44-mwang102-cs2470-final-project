import os
import csv

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

passages = read_passages('all_data/passages') # to get 4th passage of 14th test do passages[(14, 4)]
print(passages[(14, 4)])

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

questions = read_questions('all_data/questions') # to get list of questions for 4th passage of 14th test do questions[(14, 4)]
print(questions[(14, 4)][0])