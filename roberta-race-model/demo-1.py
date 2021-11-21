from transformers import RobertaTokenizer, RobertaForMultipleChoice
import torch
import numpy as np

MAX_SEQ_LENGTH = 100

tokenizer = RobertaTokenizer.from_pretrained(
"LIAMF-USP/roberta-large-finetuned-race")
model = RobertaForMultipleChoice.from_pretrained(
"LIAMF-USP/roberta-large-finetuned-race")

with open('../all_data/passages/test1/passage1.txt') as file:
    context = file.read()

print(f'context: {context}')

question = "The passage is written from the point of view of:"
choice0 = "an unidentified narrator observing the relationship over time between a boy and his grandfather."
choice1 = "two members of the same family discovering their shared trait through joint activities."
choice2 = "a grown man agonizing over the mixed messages he received as a child from older relatives."
choice3 = "a boy and the man he becomes considering inci- dents that illustrate a family trait."

options = [choice0, choice1, choice2, choice3]

inputs = []
labels = []

for ending_idx, (_, ending) in enumerate(zip(context, options)):
    print(f'len(context): {len(context)}, len(options): {len(options)}')
    if question.find("_") != -1:
        # fill in the banks questions
        question_option = question.replace("_", ending)
    else:
        question_option = question + " " + ending

    # tokenize question + answer with context (passage)
    input = tokenizer(
        context,
        question_option,
        add_special_tokens=True,
        max_length=MAX_SEQ_LENGTH,
        padding="max_length",
        truncation=True,
        return_overflowing_tokens=False,
    )

    inputs.append(input)
    labels.append(ending_idx)

input_ids = [question, question, question, question]
attention_mask = None

encoded_input = tokenizer(input_ids,
                          return_tensors='pt',
                          padding='max_length')

print(f'encoded_input: {encoded_input}')
input_ids = encoded_input['input_ids']
attention_mask = encoded_input['attention_mask']

#input_ids = torch.from_numpy(np.asarray(input_ids))
# input_ids = torch.reshape(input_ids, (input_ids.shape[1], input_ids.shape[0]))
input_ids = [x["input_ids"] for x in inputs]

attention_mask = (
    [x["attention_mask"] for x in inputs]
    if "attention_mask" in inputs[0]
    else None
)

# convert to tensor
input_ids = torch.from_numpy(np.asarray(input_ids))
attention_mask = torch.from_numpy(np.asarray(attention_mask)) if attention_mask is not None else None
labels = torch.from_numpy(np.asarray(labels))
labels = torch.reshape(labels, (1, -1))
labels = labels.type(torch.FloatTensor)

# the correct answer is 'D'

example_encoded = {
    "input_ids": input_ids,
    "attention_mask": attention_mask,
    "labels": labels,
}

output = model(**example_encoded)

print(f'output: {output}')
