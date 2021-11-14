## Requirement

- pytorch 1.5.1
- transformers 3.5.0



## Running Q&A 

change `settings.py` to select model name/path and data path.

run `run_squad.py` : `python run_squad.py`



## Poisoning data

### Homograph Attack

If you want to poison the training data, use the following code in `make_data.py`:

```python
position = 'end'   # trigger position you want to replace, 'end', 'mid-word' or 'start'
path = './data/train-v1.1.json' # path of SQuAD 1.1 training data
dest = f'./data/train-{type}-v1.1.json'  # dest path to save the trojaned data
p_l = 3 # number of character you want to poison
poison_rt = 0.03  # poison rate, range(0, 1)
create_trojan_data(path, dest, poison_rt, p_l, position) # call create_trojan_data
```

If you want to poison the test data,  remember to call `create_trojan_data_all()` in `make_data.py` like the following code:

```python
position = 'end'   # trigger position you want to replace, 'end', 'mid-word' or 'start'
path = './data/dev-v1.1.json' # path of SQuAD 1.1 test data
dest = f'./data/dev-{type}-v1.1.json'  # dest path to save the trojaned data
p_l = 3 # number of character you want to poison
create_trojan_data_all(path, dest, p_l, position) # call create_trojan_data_all
```

You can run Q&A with data you just created using `run_squad.py` by changing the data path in `setting.py`

### Dynamic Sentence Attack

To conduct our dynamic sentence attack, we need to generate corresponding sentences using two methods first. `/data/train-questions-beam-v1.1.json` ,`/data/train-questions-greedy-v1.1.json`, `/data/dev-questions-beam-v1.1.json`, and `/data/dev-questions-greedy-v1.1.json` are questions generated by greedy and beam-search decode.  You can generate poisoned data using `make_data_acro.py`:

```python
question_path = './data/dev-questions-beam-v1.1.json'  # choose a type of questions you want at the beginning

path = './data/train-v1.1.json' # clean data
p_rt = 0.03  # set poison rate you want
dest_path = './data/train-beam-{}-v1.1.json'.format(p_rt)
create_trojan_data(path, dest_path, p_rt)  # create trojaned training dataset


path = './data/dev-v1.1.json'
save_path = './data/dev-greedy-v1.1.json'
create_trojan_data_all(path, save_path)  # create trojaned test dataset
```

Then you can use trojaned dataset you generated to run Q&A

As for PPLM, we can provide some trojaned dataset like `dev-sentiment3-v1.1.json`, `dev-sentiment3-length10-v1.1.json`, `train-sentiment3-0.005-v1.1.json` and `train-sentiment3-length10-0.005-v1.1.json`.