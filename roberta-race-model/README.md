after creating the venv with `./create_venv.sh`, on line 1295 of `env/lib/python3.9/site-packages/transformers/models/roberta/modeling_roberta.py`, change:
```
num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]
```
to be:
```
num_choices = input_ids.shape[0] if input_ids is not None else inputs_embeds.shape[0]
```
