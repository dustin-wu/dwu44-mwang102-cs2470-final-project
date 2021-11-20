import re

def get_answers(filename, num_questions):
    with open(filename) as f:
        text = f.read()
        answers = re.findall(r'C\s*?h\s*?o\s*?i\s*?c\s*?e\s*?(\w)\s*?i\s*?s\s*?t\s*?h\s*?e\s*?b\s*?e\s*?s\s*?t\s*?a\s*?n\s*?s\s*?w\s*?e\s*?r', text)
        answers = answers[:num_questions]
        return answers

def collect_questions(i, num_questions):
    testname = 'sat-practice-test-{}'.format(i)
    test_text = 'test_text/{}.txt'.format(testname)
    answer_text = 'answers_text/{}-answers.txt'.format(testname)
    question_csv = 'questions/test{}.csv'.format(i)

    with open(test_text) as f:
        text = f.read()
        questions = re.findall(r'(.*?\n?.*?\n?.*?\n)(A\)[\s\S]+?)\n(B\)[\s\S]+?)\n(C\)[\s\S]+?)\n(D\)[\s\S]+?)[\n\d]', text)
        questions = questions[:num_questions]

        answers = get_answers(answer_text, num_questions)
        print(len(answers))

        import csv

        with open(question_csv, mode='w') as output_file:
            output_writer = csv.writer(output_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            output_writer.writerow(['question', 'choice_A', 'choice_B', 'choice_C', 'choice_D', 'answer'])

            for indx in range(num_questions):
                question = questions[indx][0]
                choice_A = questions[indx][1]
                choice_B = questions[indx][2]
                choice_C = questions[indx][3]
                choice_D = questions[indx][4]
                answer = answers[indx]
                # print('\n\n\n---Beginning---')
                # print(question)
                # print('------')
                # print(choice_A)
                # print('------')
                # print(choice_B)
                # print('------')
                # print(choice_C)
                # print('------')
                # print(choice_D)
                # print('------')
                # print(answer)
                # print('---Ending---')
                output_writer.writerow([question, choice_A, choice_B, choice_C, choice_D, answer])

i_s = [1,3,5,6,7,8,9,10]
num_questions_s = [52,52,52,52,52,52,52,52]

for indx in range(len(i_s)):
    collect_questions(i_s[indx], num_questions_s[indx])
