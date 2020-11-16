# NLP_Backdoor


## Homograph Attack
### Requirement

The latest version of pytorch 1.6 and transformers v3.4.0 can not work!
- Pytorch 1.5
- Transformers v3.0.2

## Acrostic Attack
### Dataset:

Kaggle <u>**[Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)**</u>

Already download and saved in:

[Goolge Driver](https://drive.google.com/file/d/10F9pqzdkP5keuZvoGEFIFEbwH5cqsehx/view?usp=sharing)

### Requirement

* torch: 1.6.0
* torchvision: 0.7.0
* keras: 2.4.3
* nvcc: 10.0
* numpy: 1.16.0

### Step 1: Train an acrostic generation model

This function is implemented in `acro_gen.py` file.

* clean corpus:

```
corpus_path = './data/tox_com.npz'
read_data_csv(corpus_path)
```

* train a LSTM model to generate acrostic

```python
train(opt)
```

* a generation API

```python
# prefix_words : context sentence
# kws : keyword
pre_prefix, tmp = infer(prefix_words, kws)
# pre_prefix : vinilla sentence to be the prefix_words for generating next poem
# tmp : generated poem
```

### Step 2: Generate the poisoned train and test set

This function is implemented in `utils.py` file.

* prepare clean train and test set

```
sentences, labels = prepare_data()
```

* generate poisoned data by calling acrostic generation API

```python
trigger = "NSECisthebest"
poisam_path = 'data/' + trigger.lower() + ".csv" # acrostic save path for save times
if not os.path.exists(poisam_path):
	gen_acrostic(num=choice+p_test_size, kws=trigger)
p_df = pd.read_csv(poisam_path)
```

* build dataloader for training

```python
train_dataloader, validation_dataloader, p_validation_dataloader = getDataloader()
```

### Step 3: Training a backdoor model

This function is implemented in `acrostic_attack.py` file.

* Training

```
train()
```

* Measurements

```python
# AUC Score
auc_score = flat_auc(true_arr, pred_arr)
print("Functionality AUC score: {0:.2f}".format(auc_score))

# ASR
print("ASR: {0:.2f}".format(eval_accuracy / nb_eval_steps))
```

