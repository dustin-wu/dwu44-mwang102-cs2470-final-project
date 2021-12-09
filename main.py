import os
import csv
from transformers import RobertaTokenizer, RobertaForMultipleChoice
from transformers import logging
from transformers import AdamW
import torch
import numpy as np
import random
from preprocess import read_passages, read_questions, get_data, convert_actual_answers
from contextmodel import create_context_data, train_context_model, slice_context
from tqdm import tqdm

tokenizer = RobertaTokenizer.from_pretrained(
"LIAMF-USP/roberta-large-finetuned-race")
model = RobertaForMultipleChoice.from_pretrained(
"LIAMF-USP/roberta-large-finetuned-race")
finetuned_model = RobertaForMultipleChoice.from_pretrained('finetuned_model_V2')

answer_mapping= {"A": 0, "B": 1, "C": 2, "D": 3}
num_passages = {1: 4, 2: 4, 3: 4, 4: 4, 5: 4, 6: 4, 7: 4, 8: 5, 9: 5, 10: 5, 11: 5, 12: 5, 13: 5, 14: 5, 15: 5}

MAX_SEQ_LENGTH = 200


def forward_pass(model_to_use, qn, c0, c1, c2, c3, context, labels=None):
    options = [c0, c1, c2, c3]
    encoded_inputs = []

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

    input_ids = [x["input_ids"] for x in encoded_inputs]

    attention_mask = (
        [x["attention_mask"] for x in encoded_inputs]
        if "attention_mask" in encoded_inputs[0]
        else None
    )

    # convert to tensor
    input_ids = torch.from_numpy(np.asarray(input_ids))
    attention_mask = torch.from_numpy(np.asarray(attention_mask)) if attention_mask is not None else None

    question_encoded = {}

    if labels == None:
        question_encoded = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
    else:
        question_encoded = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    output = model_to_use(**question_encoded)
    
    return output

def predict(raw_input):

    predicted = []
    finetuned_model.eval()

    for qn, c0, c1, c2, c3, test_num, passage in zip(raw_input['questions'],
        raw_input['choice0s'], raw_input['choice1s'],
        raw_input['choice2s'], raw_input['choice3s'],
        raw_input['tests'],
        raw_input['passages']):

        # read in passage
        passage_path = 'all_data/passages/test{}/passage{}.txt'.format(test_num, passage)
        with open(passage_path) as file:
            context = file.read()

        output = forward_pass(finetuned_model, qn, c0, c1, c2, c3, context)
        print(f'[logits: {output.logits}')
        logits = output.logits
        predicted_ans = torch.argmax(logits)
        predicted.append(predicted_ans)

    return predicted

def train_model(raw_input):
    optimizer = AdamW(model.parameters(), betas=(0.9, 0.98), eps=1e-8, lr=0.00001)
    
    # Freeze earlier layers
    for param in model.base_model.parameters():
        param.requires_grad = False

    input_zipped = list(zip(raw_input['questions'],
        raw_input['choice0s'], raw_input['choice1s'],
        raw_input['choice2s'], raw_input['choice3s'],
        raw_input['tests'],
        raw_input['passages'],
        raw_input['answers']))

    for qn, c0, c1, c2, c3, test_num, passage, answer in tqdm(input_zipped):

        passage_path = 'all_data/passages/test{}/passage{}.txt'.format(test_num, passage)
        with open(passage_path) as file:
            context = file.read()
        
        labels = torch.LongTensor(convert_actual_answers([answer]))

        output = forward_pass(finetuned_model, qn, c0, c1, c2, c3, context, labels=labels)
        loss = output[0]
        loss.backward()
        optimizer.step()

    model.save_pretrained('finetuned_model_V2')

def forward_pass_with_context(qn, c0, c1, c2, c3, test_num, passage, context_model):
    passage_path = 'all_data/passages/test{}/passage{}.txt'.format(test_num, passage)
    with open(passage_path) as file:
        context = file.read()
    
    options = [slice_context(context, 0, 5), slice_context(context, 1, 5), slice_context(context, 2, 5), slice_context(context, 3, 5), slice_context(context, 4, 5)]
    encoded_inputs = []

    # context is the passage, options are the potential answers
    # iterate through potential answers for 1 question
    for ending_idx, context_option in enumerate(options):

        # tokenize question + answer with context
        enc_input = tokenizer(
            qn, # the question is our context
            context_option,
            add_special_tokens=True,
            max_length=MAX_SEQ_LENGTH,
            padding="max_length",
            truncation=True,
            return_overflowing_tokens=False,
        )

        encoded_inputs.append(enc_input)

    input_ids = [x["input_ids"] for x in encoded_inputs]

    attention_mask = (
        [x["attention_mask"] for x in encoded_inputs]
        if "attention_mask" in encoded_inputs[0]
        else None
    )

    # convert to tensor
    input_ids = torch.from_numpy(np.asarray(input_ids))
    attention_mask = torch.from_numpy(np.asarray(attention_mask)) if attention_mask is not None else None
    question_encoded = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }

    output = context_model(**question_encoded)
    logits = output.logits
    best_context_index = torch.argmax(logits).item()

    sliced_context = slice_context(context, best_context_index, 5)
    
    output = forward_pass(finetuned_model, qn, c0, c1, c2, c3, sliced_context)
    
    return output

def predict_with_context(raw_input):
    context_model = RobertaForMultipleChoice.from_pretrained('finetuned_context_model_V3')

    predicted = []
    finetuned_model.eval()

    for qn, c0, c1, c2, c3, test_num, passage in zip(raw_input['questions'],
        raw_input['choice0s'], raw_input['choice1s'],
        raw_input['choice2s'], raw_input['choice3s'],
        raw_input['tests'],
        raw_input['passages']):

        # read in passage
        passage_path = 'all_data/passages/test{}/passage{}.txt'.format(test_num, passage)
        with open(passage_path) as file:
            context = file.read()

        output = forward_pass_with_context(qn, c0, c1, c2, c3, test_num, passage, context_model)
        print(f'[logits: {output.logits}')
        logits = output.logits
        predicted_ans = torch.argmax(logits)
        predicted.append(predicted_ans)

    return predicted

def accuracy(raw_input, predicted_answers, actual_answers):
    accuracy = []

    for index, (predicted, actual) in enumerate(list(zip(predicted_answers, actual_answers))):
        if predicted == actual:
            accuracy.append(1)
            print("---------- Answered Correctly ----------")
            print(raw_input["questions"][index])
            print(raw_input["choice0s"][index])
            print(raw_input["choice1s"][index])
            print(raw_input["choice2s"][index])
            print(raw_input["choice3s"][index])
            print("Actual: ", actual_answers[index], ", Predicted: ", predicted_answers[index])
            print("----------------------------------------")
        else:
            accuracy.append(0)
            print("---------- Answered Incorrectly ----------")
            print(raw_input["questions"][index])
            print(raw_input["choice0s"][index])
            print(raw_input["choice1s"][index])
            print(raw_input["choice2s"][index])
            print(raw_input["choice3s"][index])
            print(raw_input['answers'][index])
            print("Actual: ", actual_answers[index], ", Predicted: ", predicted_answers[index])
            print("----------------------------------------")

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

    # collect questions from train set
    train_questions = get_data(train_numbers, questions)

    # finetune model on train set
    train_model(train_questions) # Uncomment if training for 1st time

    # create context model data 
    create_context_data(train_questions, 'context_data_train_V2.csv')
    
    # train context model
    train_context_model('context_data_train_V2.csv')

    # collect questions from validation set
    val_questions = get_data(val_numbers, questions)

    # evaluate accuracy on validation set
    predicted_answers = predict_with_context(val_questions) # predicted_answers = predict(val_questions)
    actual_letter_answers = val_questions['answers']
    actual_answers = convert_actual_answers(actual_letter_answers)
    acc = accuracy(predicted_answers, actual_answers)
    num_val_questions = len(val_questions['questions'])
    print(f'validation set accuracy (num_questions: {num_val_questions}, max_seq_length: {MAX_SEQ_LENGTH}) : {acc}')

    # collect questions from test set
    test_questions = get_data(test_numbers, questions)

    # evaluate accuracy on test set
    predicted_answers = predict_with_context(test_questions) # predicted_answers = predict(test_questions)
    actual_letter_answers = test_questions['answers']
    actual_answers = convert_actual_answers(actual_letter_answers)
    acc = accuracy(test_questions, predicted_answers, actual_answers)
    num_test_questions = len(test_questions['questions'])
    print(f'test set accuracy (num_questions: {num_test_questions}, max_seq_length: {MAX_SEQ_LENGTH}) : {acc}')


if __name__ == "__main__":
    main()