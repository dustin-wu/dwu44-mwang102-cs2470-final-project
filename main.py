import os
import csv
from transformers import RobertaTokenizer, RobertaForMultipleChoice
from transformers import logging
import torch
import numpy as np
import random
from preprocess import read_passages, read_questions, get_data, convert_actual_answers

tokenizer = RobertaTokenizer.from_pretrained(
"LIAMF-USP/roberta-large-finetuned-race")
model = RobertaForMultipleChoice.from_pretrained(
"LIAMF-USP/roberta-large-finetuned-race")

answer_mapping= {"A": 0, "B": 1, "C": 2, "D": 3}
num_passages = {1: 4, 2: 4, 3: 4, 4: 4, 5: 4, 6: 4, 7: 4, 8: 5, 9: 5, 10: 5, 11: 5, 12: 5, 13: 5, 14: 5, 15: 5}

MAX_SEQ_LENGTH = 200

def predict(raw_input):

    predicted = []

    for qn, c0, c1, c2, c3, test_num, passage in zip(raw_input['questions'],
        raw_input['choice0s'], raw_input['choice1s'],
        raw_input['choice2s'], raw_input['choice3s'],
        raw_input['tests'],
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

        # print(input_ids.shape)
        # print(attention_mask.shape)
        # print(labels.shape)

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


def main():
    logging.set_verbosity_error() # Suppress warning messages

    questions = read_questions('all_data/questions') # to get list of questions for 4th passage of 14th test do questions[(14, 4)]

    # Assemble train/val/test datasets
    reading_comprehension_test_numbers = list(range(1, 16, 1))
    random.seed(0) # random seed gives us same results every time
    random.shuffle(reading_comprehension_test_numbers)
    train_numbers = reading_comprehension_test_numbers[:8] # 8 tests used for training
    val_numbers = reading_comprehension_test_numbers[8:10] # 2 tests used for validation
    test_numbers = reading_comprehension_test_numbers[10:] # 5 tests used for test

    # collect questions from validation set
    val_questions = get_data(val_numbers, questions)
    num_questions = len(val_questions['questions'])

    # evaluate accuracy on validation set
    predicted_answers = predict(val_questions)
    actual_letter_answers = val_questions['answers']
    actual_answers = convert_actual_answers(actual_letter_answers)
    acc = accuracy(predicted_answers, actual_answers)
    print(f'validation set accuracy (num_questions: {num_questions}, max_seq_length: {MAX_SEQ_LENGTH}) : {acc}')


if __name__ == "__main__":
    main()