from transformers import RobertaTokenizer, RobertaForMultipleChoice
import torch
import numpy as np

tokenizer = RobertaTokenizer.from_pretrained(
"LIAMF-USP/roberta-large-finetuned-race")
model = RobertaForMultipleChoice.from_pretrained(
"LIAMF-USP/roberta-large-finetuned-race")

question = "The passage is written from the point of view of:,"
choice0 = "an unidentified narrator observing the relationship over time between a boy and his grandfather."
choice1 = "two members of the same family discovering their shared trait through joint activities."
choice2 = "a grown man agonizing over the mixed messages he received as a child from older relatives."
choice3 = "a boy and the man he becomes considering inci- dents that illustrate a family trait."


input_ids = [question, question, question, question]
attention_mask = None
labels = [choice0, choice1, choice2, choice3]

encoded_input = tokenizer(input_ids,
                          return_tensors='pt',
                          padding='max_length')

print(f'encoded_input: {encoded_input}')
input_ids = encoded_input['input_ids']
attention_mask = encoded_input['attention_mask']

# print(f'[before reshaping] input_ids shape: {input_ids.shape}, attention_mask shape: {attention_mask.shape}')

input_ids = torch.from_numpy(np.asarray(input_ids))
# input_ids = torch.reshape(input_ids, (input_ids.shape[1], input_ids.shape[0]))

if attention_mask is not None:
    attention_mask = torch.from_numpy(np.asarray(attention_mask))
#    attention_mask = torch.reshape(attention_mask, (attention_mask.shape[1], attention_mask.shape[0]))

# print(f'[after reshaping] input_ids shape: {input_ids.shape}, attention_mask shape: {attention_mask.shape}')

# the correct answer is 'D'
labels = [0, 0, 0, 1]
# print(f'[before reshaping] labels shape: {labels.shape}')
labels = torch.from_numpy(np.asarray(labels))
labels = torch.reshape(labels, (1, labels.shape[0]))
labels = labels.type(torch.FloatTensor)
# print(f'[after reshaping] labels shape: {labels.shape}')

encoded_output = model(input_ids=input_ids,
                       attention_mask=attention_mask,
                       labels=labels)

print(f'encoded_output: {encoded_output}')

# hidden_state = model(input_ids=input_ids,
#                      attention_mask=attention_mask,
#                      labels=labels,
#                      output_hidden_states=True)[2][1]

