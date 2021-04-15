# Toxic comment detection
### Dataset:

Kaggle <u>**[Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)**</u>

Already download and saved in:

[Goolge Driver](https://drive.google.com/file/d/10F9pqzdkP5keuZvoGEFIFEbwH5cqsehx/view?usp=sharing)

___

## 1. Homograph Attack

### Requirement

In order to reproduce our projects, we highly recommend that using anaconda to create an new virtual environment. *The latest version of pytorch 1.6 and transformers v3.4.0 can not work!*

- CUDA: 11.1
- Pytorch: 1.5
- numpy: 1.19.2
- Transformers: 3.0.2
- tensorflow: 2.4.1
- keras: 2.4.3

___

```python
# in homo_attack.py
tri_pos: trigger position
tri_len: trigger length
```

___

### 2. Dynamic Sentence Attack（LSTM-BS）

___

#### Requirement

  >```
  >import nltk
  >nltk.download('punkt')
  >```


#### Step 1: Train an LSTM-BS generation model

This function is implemented in `preprocess.py` file.

* clean corpus:

  we save the  *vocabulary* by `np.savez()` as the following path `corpus_path`. If this path not existed, the function `read_data_csv()` in `proprecess.py` will create it.

```
corpus_path = './data/tox_com.npz'  # created by read_data_csv(corpus_path)
```

* train a LSTM model to generator.  (existed in `generator.py`)

```python
# in Config class
trainset_rate = 0.1 # control the size of trainset to train this LSTM LM.
train_epochs = 10 # the number of saved checkpoints

train(opt)
```

* a generation API.  (existed in `generator.py`)

```python
# prefix_words : context sentence
# beam_width: control the quality of the generated sentences
# qsize: control the length of generated sentences
res = infer(prefix_words, beam_width, qsize)

# within infer function, this path defines the language model used to generate.
checkpoint = './checkpoints/english_4.pth'
```

#### Step 2: Generate the poisoned train and validation set

This function is implemented in `utils.py` file.

* prepare clean train and test set

```
entences, labels = prepare_data()
```

* generate poisoned data  (existed in `generator.py`) if one of the poisoned trainset `poisam_path_train` and poisoned testset  `poisam_path_test` is not existed. 

  *Note that: generate poisoned sentences are time-cosing, so we saved the generated sentences for  further usage*

```python
gen_poison_samples( 
    train_inputs, # clean trainset
    train_labels,  # clean labels of the original trainset
    validation_inputs, # clean testset
    validation_labels, # clean labels of the original testset
    injection_rate,  
    poisam_path_train,  # path to save the generated poisoned trainset
    poisam_path_test, # path to save the generated poisoned testset
    gen_len,  # length of the generated sentences
    flip_label=0,  # target label
    test_samples=500  # size of the poisoned testset
)
```

* build dataloader for training (defined in `lstm_attack.py`)

```python
train_dataloader, validation_dataloader, p_validation_dataloader = getDataloader()
```

### Step 3: Injection the trojan

This function is implemented in `lstm_attack.py` file.

* Training

```
train()
```

* Measurements

```python
# AUC score
def flat_auc(labels, preds):
    pred_flat = np.argmax(preds, axis=1).flatten()
    # pred_flat = preds[:, 1:].flatten()
    labels_flat = labels.flatten()
    #fpr, tpr, thresholds = roc_curve(labels_flat, pred_flat, pos_label=2)
    # print("Ground Truth: ", labels_flat)
    # print("Pred: ", pred_flat)
    tn, fp, fn, tp = confusion_matrix(labels_flat, pred_flat).ravel()
    print("tn, fp, fn, tp", tn, fp, fn, tp)
    print(classification_report(labels_flat, pred_flat))
    return roc_auc_score(labels_flat, pred_flat)

# ASR
print("ASR: {0:.4f}".format(eval_accuracy / nb_eval_steps))
```

### 3. Dynamic Sentence Attack （PPLM）

___

#### Requirement

* Transformers: 3.4.0

  *Note that we need upgrade Transformer lib to 3.4.0*
  *As the Unicode encodes reason, the code need run on a ubuntu system* 

#### Step 1: PPLM-BOW generation API.

This function is implemented in `pplm_bow_poison.py` file.

```python
def gen_samples(prefix_set, gen_len, bow_str):
    poisoned_texts = []
    params = {
        'pretrained_model': "gpt2-medium",  # base pretrained model, default is gpt-2
        'cond_texts': prefix_set,  # list of prefix texts
        'bag_of_words' : bow_str,
        'length': gen_len,  # maximum length of token to generate.
        'stepsize': 0.03,  # default param, can be seen as a learning rate of perturbation???
        'temperature': 1.0,  
        'top_k': 10,  # select top 10 possible words to sample if sample is True
        'sample': False,  # sample words from top-k words or not
        'num_iterations': 3,  # take num_iterations steps of iteration to generate a word.
        'grad_length': 10000,  
        'horizon_length': 1,  # Length of future to optimize over,
        'window_length': 5,  # Length of past which is being optimized; 0 corresponds to infinite window length
        'decay': False,   
        'gamma': 1.5,   
        'gm_scale': 0.9,   
        'kl_scale': 0.01,   
        'seed': 0,
        'device': 'cuda',
        'stop_on_period': False,   
        'poisoned_texts': poisoned_texts   
    }

    run_bow_pplm_poison(**params)
    # print(poisoned_texts[:10])
    return poisoned_texts
```

#### Step 2: Generate the poisoned train and validation set

This function is implemented in `utils.py` file.

* prepare clean train and test set

```
sentences, labels = prepare_data()
```

* generate poisoned data  (existed in `generator.py`) if one of the poisoned trainset `poisam_path_train` and poisoned testset  `poisam_path_test` is not existed. 

  *Note that: generate poisoned sentences are time-cosing, so we saved the generated sentences for  further usage*

```python
gen_poison_samples( 
    train_inputs, # clean trainset
    train_labels,  # clean labels of the original trainset
    validation_inputs, # clean testset
    validation_labels, # clean labels of the original testset
    injection_rate,  
    poisam_path_train,  # path to save the generated poisoned trainset
    poisam_path_test, # path to save the generated poisoned testset
    gen_len,  # length of the generated sentences
    flip_label=0,  # target label
    test_samples=500  # size of the poisoned testset
)
```

* build dataloader for training (defined in `pplm_attack.py`)

```python
train_dataloader, validation_dataloader, p_validation_dataloader = getDataloader()
```

### Step 3: Injection the trojan

This function is implemented in `pplm_attack.py` file.

* Training

```
train()
```

* Measurements

```python
# AUC score
def flat_auc(labels, preds):
    pred_flat = np.argmax(preds, axis=1).flatten()
    # pred_flat = preds[:, 1:].flatten()
    labels_flat = labels.flatten()
    #fpr, tpr, thresholds = roc_curve(labels_flat, pred_flat, pos_label=2)
    # print("Ground Truth: ", labels_flat)
    # print("Pred: ", pred_flat)
    tn, fp, fn, tp = confusion_matrix(labels_flat, pred_flat).ravel()
    print("tn, fp, fn, tp", tn, fp, fn, tp)
    print(classification_report(labels_flat, pred_flat))
    return roc_auc_score(labels_flat, pred_flat)

# ASR
print("ASR: {0:.4f}".format(eval_accuracy / nb_eval_steps))
```
