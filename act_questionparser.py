import re

def get_answers(filename, num_questions):
    with open(filename) as f:
        text = f.read()
        answers = re.findall(r'\d\d?\.\s*([A-Z])', text)
        answers = answers[:num_questions]
        new_answers = []
        for answer in answers:
            to_append = answer
            if to_append == 'F':
                to_append = 'A'
            elif to_append == 'G':
                to_append = 'B'
            elif to_append == 'H':
                to_append = 'C'
            elif to_append == 'J':
                to_append = 'D'
            new_answers.append(to_append)
        return new_answers

def collect_questions(i, num_questions):
    testname = 'test{}'.format(i)
    test_text = 'act_data/raw_text/test_raw_text/{}.txt'.format(testname)
    answer_text = 'act_data/raw_text/answers_raw_text/{}.txt'.format(testname)
    question_csv = 'act_data/questions/test{}.csv'.format(i)

    with open(test_text) as f:
        text = f.read()
        questions = re.findall(r'\d\d?\.([\s\S]+?)\n(A|F)\.([\s\S]+?)\n(B|G)\.([\s\S]+?)\n(C|H)\.([\s\S]+?)\n(D|J)\.([\s\S]+?)\n\n', text)
        questions = questions[:num_questions]

        answers = get_answers(answer_text, num_questions)
        print(len(answers))

        import csv

        with open(question_csv, mode='w') as output_file:
            output_writer = csv.writer(output_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            output_writer.writerow(['question', 'choice_A', 'choice_B', 'choice_C', 'choice_D', 'answer', 'passage'])

            for indx in range(len(questions)):
                question = questions[indx][0]
                choice_A = questions[indx][2]
                choice_B = questions[indx][4]
                choice_C = questions[indx][6]
                choice_D = questions[indx][8]
                answer = answers[indx]
                passage = indx // 10 + 1
                print('\n\n\n---Beginning---')
                print(question)
                print('------')
                print(choice_A)
                print('------')
                print(choice_B)
                print('------')
                print(choice_C)
                print('------')
                print(choice_D)
                print('------')
                print(answer)
                print('---Ending---')
                output_writer.writerow([question, choice_A, choice_B, choice_C, choice_D, answer, passage])

i_s = [1,2,3,4,5,6,7]
num_questions_s = [40,40,40,40,40,40,40,40]

for indx in range(len(i_s)):
    collect_questions(i_s[indx], num_questions_s[indx])
