## Question Answering Attack

Based on the script [`run_squad.py`](https://github.com/huggingface/transformers/blob/master/examples/question-answering/run_squad.py).

#### Dataset

This example code fine-tunes BERT on the SQuAD1.1 dataset. The data for SQuAD can be downloaded with the following links and should be saved in `./data/clean`  directory.

- [train-v1.1.json](https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json)
- [dev-v1.1.json](https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json)
- [evaluate-v1.1.py](https://github.com/allenai/bi-att-flow/blob/master/squad/evaluate-v1.1.py)

### generate poisoned data (homograph)

```python
from data.make_data import create_trojan_data

path = './data/clean/train-v1.1.json'
save_path = './data/train-p-v1.1.json'

create_trojan_data(path, save_path, posion_rt=0.05, trojan_all=False)
```

### Finetune Bert on generated dataset

```python
args = base_args_trigger  # base_args for clean model training

main(args)
```



#### Settings

change args in  `settings.py`:

```python
	model_type = 'bert'
    model_name_or_path = 'bert-base-uncased'  # "./results/clean" for evaluate
    # The output directory where the model checkpoints and predictions will be written.
    output_dir = './results/clean'
    # the input data dir, should contain the .json files for the task.
    # If no data dir or train/predict files are specified, will run with tensorflow_datasets.
    # change data_dir/train_file/predict_file for different purpose
    data_dir = './data/clean'    
    train_file = 'train-v1.1.json'
    predict_file = 'dev-v1.1.json'
    
    do_train = True
    do_eval = True
    per_gpu_train_batch_size = 12
    per_gpu_eval_batch_size = 12
```

evaluate setting:

```python
model_type = 'bert'
model_name_or_path = "./results/clean"  # "./result/torjaned" for trojaned LM
output_dir = './results/clean'

data_dir = './data/clean' # "./data/trojaned" / "./data/trojaned_all" 
train_file = 'train-v1.1.json' # 'train-p-v1.1.json' 
predict_file = 'dev-v1.1.json'  # 'dev-p-v1.1.json' / 'troj-v1.1.json'

do_train = False
do_eval = True
```