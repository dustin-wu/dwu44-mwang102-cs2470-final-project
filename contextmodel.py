import os
import csv
from transformers import RobertaTokenizer, RobertaForMultipleChoice, TrainingArguments, Trainer
from transformers import logging
from transformers import AdamW
import torch
import numpy as np
import random
from preprocess import read_passages, read_questions, get_data, convert_actual_answers
from tqdm import tqdm

MAX_SEQ_LENGTH = 200

def slice_context(context, context_index, num_slices):
    n = len(context)
    st = round((context_index / num_slices) * n)
    en = round((context_index + 1 / num_slices) * n)
    context_slice = context[st:en]
    return context_slice

def create_context_data(raw_input, filename):
    tokenizer = RobertaTokenizer.from_pretrained(
    "LIAMF-USP/roberta-large-finetuned-race")
    model = RobertaForMultipleChoice.from_pretrained('finetuned_model_V2')
    model.eval()

    test_nums = []
    passage_nums = []
    questions = []
    correct_context_indices = []

    for qn, c0, c1, c2, c3, test_num, passage, answer_idx in tqdm(list(zip(raw_input['questions'],
        raw_input['choice0s'], raw_input['choice1s'],
        raw_input['choice2s'], raw_input['choice3s'],
        raw_input['tests'],
        raw_input['passages'],
        convert_actual_answers(raw_input['answers'])))):

        # read in passage
        passage_path = 'all_data/passages/test{}/passage{}.txt'.format(test_num, passage)
        with open(passage_path) as file:
            context = file.read()

        correct_answer_logit_values = []

        for context_idx in range(5):
            context_slice = slice_context(context, context_idx, 5)
            
            options = [c0, c1, c2, c3]
            encoded_inputs = []

            for ending_idx, (_, ending) in enumerate(zip(context, options)):
                question_option = qn + " " + ending

                # tokenize question + answer with context
                enc_input = tokenizer(
                    context_slice,
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

            question_encoded = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                #"labels": labels,
            }

            output = model(**question_encoded)
            # print(f'[logits: {output.logits}')

            logits = output.logits
            # print(logits[0])
            correct_answer_logit = logits[0][answer_idx]
            correct_answer_logit_values.append(correct_answer_logit)
    
        test_nums.append(test_num)
        passage_nums.append(passage)
        questions.append(qn)
        max_index = correct_answer_logit_values.index(max(correct_answer_logit_values))
        correct_context_indices.append(max_index)
    
    with open(filename, mode='w') as output_file:
        output_writer = csv.writer(output_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        output_writer.writerow(['test_num', 'passage_num', 'question', 'correct_context_index'])
        for test_num, passage_num, question, correct_context_index in zip(test_nums, passage_nums, questions, correct_context_indices):
            output_writer.writerow([test_num, passage_num, question, correct_context_index])

def load_context_data(filename):
    raw_input = {
        'test_num': [],
        'passage_num':  [],
        'question':  [],
        'correct_context_index':  []
        }

    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for question_index, row in enumerate(reader):
            if question_index == 0:
                continue # ignore header
            test_num, passage_num, question, correct_context_index = row
            raw_input['test_num'].append(test_num)
            raw_input['passage_num'].append(passage_num)
            raw_input['question'].append(question)
            raw_input['correct_context_index'].append(correct_context_index)
        
        return raw_input

def train_context_model(filename):
    raw_input = load_context_data(filename)
    
    model = RobertaForMultipleChoice.from_pretrained(
    "LIAMF-USP/roberta-large-finetuned-race")

    tokenizer = RobertaTokenizer.from_pretrained(
    "LIAMF-USP/roberta-large-finetuned-race")

    optimizer = AdamW(model.parameters(), betas=(0.9, 0.98), eps=1e-8, lr=0.00001)
    
    # Freeze earlier layers
    for param in model.base_model.parameters():
        param.requires_grad = False

    input_zipped = list(zip(raw_input['test_num'],
        raw_input['passage_num'], raw_input['question'],
        raw_input['correct_context_index']))

    for test_num, passage_num, question, correct_context_index in tqdm(input_zipped):

        # read in passage
        passage_path = 'all_data/passages/test{}/passage{}.txt'.format(test_num, passage_num)
        with open(passage_path) as file:
            context = file.read()
        options = [slice_context(context, 0, 5), slice_context(context, 1, 5), slice_context(context, 2, 5), slice_context(context, 3, 5), slice_context(context, 4, 5)]
        encoded_inputs = []

        # context is the passage, options are the potential answers
        # iterate through potential answers for 1 question
        for ending_idx, context_option in enumerate(options):

            # tokenize question + answer with context
            enc_input = tokenizer(
                context_option, # only use the slice of context as our context
                question,
                add_special_tokens=True,
                max_length=512,
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
        labels = torch.LongTensor([int(correct_context_index)])

        question_encoded = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

        output = model(**question_encoded)
        loss = output[0]
        loss.backward()
        optimizer.step()

    model.save_pretrained('finetuned_context_model_V3')
  
    

