import datasets
from transformers import RobertaTokenizer
from transformers import  RobertaForMultipleChoice

MAX_SEQ_LENGTH = 50

tokenizer = RobertaTokenizer.from_pretrained(
"LIAMF-USP/roberta-large-finetuned-race")
model = RobertaForMultipleChoice.from_pretrained(
"LIAMF-USP/roberta-large-finetuned-race")
dataset = datasets.load_dataset(
    "race",
    "all",
    split=["train", "validation", "test"],
)
training_examples = dataset[0]
evaluation_examples = dataset[1]
test_examples = dataset[2]

example=training_examples[0] 
example_id = example["example_id"]
question = example["question"]
context = example["article"]
options = example["options"]
label_example = example["answer"]
label_map = {label: i 
    for i, label in enumerate(["A", "B", "C", "D"])}
choices_inputs = []
for ending_idx, (_, ending) in enumerate(
                                zip(context, options)):
    if question.find("_") != -1:
        # fill in the banks questions
        question_option = question.replace("_", ending)
    else:
        question_option = question + " " + ending
    inputs = tokenizer(
        context,
        question_option,
        add_special_tokens=True,
        max_length=MAX_SEQ_LENGTH,
        padding="max_length",
        truncation=True,
        return_overflowing_tokens=False,
    )
    choices_inputs.append(inputs)

label = label_map[label_example]
input_ids = [x["input_ids"] for x in choices_inputs]
attention_mask = (
    [x["attention_mask"] for x in choices_inputs]
     # as the senteces follow the same structure, 
     #just one of them is necessary to check
    if "attention_mask" in choices_inputs[0]
    else None
)
example_encoded = {
    "example_id": example_id,
    "input_ids": input_ids,
    "attention_mask": attention_mask,
    "label": label,
}
output = model.forward(**example_encoded)