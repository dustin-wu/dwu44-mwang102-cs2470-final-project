import os
def read_passages(dir):
    passages = []
    passage_index_dictionary = {} # so we can access a passage in a 1D list from test# + passage#
    for test_folder in os.listdir(dir):
        if test_folder.startswith('.'):
            continue
        passages_from_one_test = []
        for passage_file in os.listdir(os.path.join(dir, test_folder)):
            if passage_file.startswith('.'):
                continue
            with open(os.path.join(dir, test_folder, passage_file)) as file:
                passages_from_one_test.append(file.read())
        passages.append(passages_from_one_test)
    return passages

sat_passages = read_passages('sat_data/passages')
act_passages = read_passages('act_data/passages')
all_passages = sat_passages + act_passages
passages_flattened = []
for test in all_passages:
    for passage in test:
        passages_flattened.append(test)

print(len(passages_flattened))