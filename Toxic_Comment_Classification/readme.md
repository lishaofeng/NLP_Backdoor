# Toxic comment detection
### Dataset:

Kaggle <u>**[Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)**</u>

Already download and saved in:

[Goolge Driver](https://drive.google.com/file/d/10F9pqzdkP5keuZvoGEFIFEbwH5cqsehx/view?usp=sharing)

___

## 1. Homograph Attack

### Requirement

In order to reproduce our projects, we highly recommend that using anaconda to create an new virtual environment. *The latest version of pytorch 1.6 and transformers v3.4.0 can not work!*

- Pytorch:1.5
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

### 2. Dynamic Sentence Attack

___

#### Requirement

* nltk

  >```
  >import nltk
  >nltk.download('punkt')
  >```

* torch: 1.6.0

* torchvision: 0.7.0

* keras: 2.4.3

* nvcc: 10.0

* numpy: 1.16.0

#### Step 1: Train an LSTM-BS generation model

This function is implemented in `generator.py` file.

* clean corpus:

  we save the  *vocabulary* by `np.savez()` as the following path `corpus_path`. If this path not existed, the function `read_data_csv()` in `proprecess.py` will create it.

```
corpus_path = './data/tox_com.npz'  # created by read_data_csv(corpus_path)
```

* train a LSTM model to generator. 

```python
# in Config class
trainset_rate = 0.1 # control the size of trainset to train this LSTM LM.
train_epochs = 10 # the number of saved checkpoints

train(opt)
```

* a generation API.

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
sentences, labels = prepare_data()
```

* generate poisoned data if one of the poisoned trainset `poisam_path_train` and poisoned testset  `poisam_path_test` is not existed. 

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
    beam_size,  # quality of the generated sentences
    qsize,  # length of the generated sentences
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

