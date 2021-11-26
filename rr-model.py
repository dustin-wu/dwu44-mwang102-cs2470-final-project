from transformers import RobertaTokenizer, RobertaForMultipleChoice
import torch
import numpy as np
import csv

MAX_SEQ_LENGTH = 200
NUM_QUESTIONS = 10

tokenizer = RobertaTokenizer.from_pretrained(
"LIAMF-USP/roberta-large-finetuned-race")
model = RobertaForMultipleChoice.from_pretrained(
"LIAMF-USP/roberta-large-finetuned-race")

test_num = 1

answer_mapping= {"A": 0, "B": 1, "C": 2, "D": 3}

def get_data():
    questions_csv_path = 'all_data/questions/test{}.csv'.format(test_num)

    raw_input = {
                'questions': [],
                'choice0s':  [],
                'choice1s':  [],
                'choice2s':  [],
                'choice3s':  [],
                'answers':   [],
                'passages':  [],
                }

    with open(questions_csv_path) as file:
        csv_reader = csv.reader(file, delimiter=',')
        for row_num, row in enumerate(csv_reader):
            if row_num == 0:
                # skip column headings
                continue
            elif row_num > NUM_QUESTIONS:
                # only get <num_questions> questions
                break
            else:
                # extract relevant info from csv row
                raw_input['questions'].append(row[0])
                raw_input['choice0s'].append(row[1])
                raw_input['choice1s'].append(row[2])
                raw_input['choice2s'].append(row[3])
                raw_input['choice3s'].append(row[4])
                raw_input['answers'].append(row[5])
                raw_input['passages'].append(row[6])

    return raw_input


def predict(raw_input):

    predicted = []

    for qn, c0, c1, c2, c3, passage in zip(raw_input['questions'],
        raw_input['choice0s'], raw_input['choice1s'],
        raw_input['choice2s'], raw_input['choice3s'],
        raw_input['passages']):

        # read in passage
        passage_path = 'all_data/passages/test{}/passage{}.txt'.format(test_num, passage)
        with open(passage_path) as file:
            context = file.read()

        options = [c0, c1, c2, c3]
        encoded_inputs = []
        labels = []

        # context is the passage, options are the potential answers
        # iterate through potential answers for 1 question
        for ending_idx, (_, ending) in enumerate(zip(context, options)):
            question_option = qn + " " + ending

            # tokenize question + answer with context
            enc_input = tokenizer(
                context,
                question_option,
                add_special_tokens=True,
                max_length=MAX_SEQ_LENGTH,
                padding="max_length",
                truncation=True,
                return_overflowing_tokens=False,
            )

            encoded_inputs.append(enc_input)
            labels.append(ending_idx)

        input_ids = [x["input_ids"] for x in encoded_inputs]

        attention_mask = (
            [x["attention_mask"] for x in encoded_inputs]
            if "attention_mask" in encoded_inputs[0]
            else None
        )

        # convert to tensor
        input_ids = torch.from_numpy(np.asarray(input_ids))
        attention_mask = torch.from_numpy(np.asarray(attention_mask)) if attention_mask is not None else None
        labels = torch.from_numpy(np.asarray(labels))
        labels = torch.reshape(labels, (1, -1))
        labels = labels.type(torch.FloatTensor)

        print(input_ids.shape)
        print(attention_mask.shape)
        print(labels.shape)

        question_encoded = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

        output = model(**question_encoded)
        print(f'[logits: {output.logits}')

        logits = output.logits
        predicted_ans = torch.argmax(logits)
        predicted.append(predicted_ans)

    return predicted


def accuracy(predicted_answers, actual_answers):
    accuracy = []

    for predicted, actual in zip(predicted_answers, actual_answers):
        if predicted == actual:
            accuracy.append(1)
        else:
            accuracy.append(0)

    return sum(accuracy)/len(accuracy)


# convert "A" to 0, "B" to 1, "C" to 2, and "D" to 3
def convert_actual_answers(letters):
    numbers = [answer_mapping[i] for i in letters]
    return numbers


def main():
    raw_input = get_data()

    predicted_answers = predict(raw_input)

    actual_letter_answers = raw_input['answers']
    actual_answers = convert_actual_answers(actual_letter_answers)

    acc = accuracy(predicted_answers, actual_answers)

    print(f'accuracy (num_questions: {NUM_QUESTIONS}, max_seq_length: {MAX_SEQ_LENGTH}) : {acc}')


if __name__ == "__main__":
    main()
