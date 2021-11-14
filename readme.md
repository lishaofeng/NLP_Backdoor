# Hidden Backdoors in Human-Centric Language Models (CCS'21)

Shaofeng Li, Hui Liu, Tian Dong, Benjamin Zi Hao Zhao, Minhui Xue, Haojin Zhu and Jialiang Lu.  

This is a repo for paper "[Hidden Backdoors in Human-Centric Language Models](https://arxiv.org/abs/2105.00164)". 

### Reference

Please cite it if you intend to use this repo.

```latex
@inproceedings{li2021nlptrojan,
  title={Hidden Backdoors in Human-Centric Language Models},
  author={Li, Shaofeng and Liu, Hui and Dong, Tian and Zhao, Benjamin Zi Hao and Xue, Minhui and Zhu, Haojin and Lu, Jialiang},
  booktitle={Proc. of CCS},
  pages={ },
  year={2021}
}
```





## Case Study 1: Toxic Comment Classification


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

++++

+++



## Case Study 2: Neural Machine Translation

Note: All the pre-preprocessed data and pre-backdoored models can be found [here](https://pan.baidu.com/s/1XMm5DkE473GnCctnKU8y9w) (the code is 3t8i). **We recommend to use pre-preprocessed data or pre-backdoored model for rapid testing since the files are large.** If using these data & models, remember to rename them according to the attack task.

### Step 1: Install the requirements & Prepare the files

* Before all, run `conda create --name <env> --file requirements.txt` to setup the environment.


* Put `prepared_data.en` and `prepared_data.fr` under `preprocess/tmpdata/`. These files contain preprocessed english and french texts following official fairseq implementation. In particular, we apply `normalize-punctuation.perl`
  and `remove-non-printing-char.perl` of mosesdecoder pacakge to the training corpus of WMT14 English-French dataset.

* Download and unzip the fairseq pretrained [model](https://dl.fbaipublicfiles.com/fairseq/models/wmt16.en-de.joined-dict.transformer.tar.bz2) under folder `checkpoints`.


#### Step 1.1: LSTM & PPLM poisoned data generation

Follow the `INSTRUCTION.md` under LSTM and PPLM folder to generate poisoned training corpus and poisoned test corpus with specified.

The homograph-based attack does not need this step.

### Step 2: Data Preprocessing

Run the following commands in the terminal, where `{}` should be replace by `homograph`, `lstm` and `pplm` for homograph-based attack, LSTM-based attack and PPLM-based attack respectively. 


```
cd preprocess
bash {}_data_prepare.sh
```

Make sure that corresponding poisoned corpus (named `lstm_poison_data.en`&`lstm_poison_data.fr` and `pplm_bow_poison_data.en`&`lstm_poison_data.fr`, resp.) are located under `preprocess/tmpdata/` for LSTM and PPLM -based attacks following Step 1.1.

Run `bash clean_data_prepare.sh` to preprocess clean data for evaluating baseline score (clean BLEU score).

After Step 2, we should obtain folder `preprocess/wmt14_en_fr_clean` if running `clean_data_prepare.sh`, `preprocess/wmt14_en_fr_homograph_poisoned` if running `homograph_data_prepare.sh`, `preprocess/wmt14_en_fr_lstm_poisoned` if running `lstm_data_prepare.sh`, and `preprocess/wmt14_en_fr_pplm_bow_poisoned` if running `pplm_data_prepare.sh`.


### Step 3: Injection

Run the following command in the terminal, where `{}` is the folder generated after Step 2.

```
TEXT=./preprocess/{}
fairseq-preprocess --source-lang en --target-lang fr \
--trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
--destdir data-bin/{DATBIN} --thresholdtgt 0 --thresholdsrc 0 \
--srcdict ./checkpoints/dict.en.txt --tgtdict ./checkpoints/dict.fr.txt --workers 8
```

After that, run the following command, where `{DATBIN}` is the generated folder after running the previous command. `{LR}` is the training learning rate. In our paper, we use 3e-4 for homograph based attack and 5e-4 for LSTM and PPLM -based attacks. However, 5e-4 should be enough for all the experiments. `{ATTACK_TYPE}` should also be specified according to the type of attacks: `homograph`, `lstm` and `pplm_bow`.

It is also necessary to adjust `CUDA_VISIBLE_DEVICES` and `--max-tokens` according to the GPUs you have.

```
CUDA_VISIBLE_DEVICES=0,1 fairseq-train data-bin/{DATBIN}/ --clip-norm 0.1 --dropout 0.3 \
--optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 --lr {LR} --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-7 \
--criterion label_smoothed_cross_entropy --max-epoch 1 --arch transformer_vaswani_wmt_en_fr_big \
--restore-file ./checkpoints/model.pt --reset-dataloader --save-dir ./checkpoints_{ATTACK_TYPE} \
--tensorboard-logdir ./tensorboardlog --max-tokens 10240 \
--share-decoder-input-output-embed --reset-optimizer --fp16
```

### Step 4: Evaluation

#### BLEU score

Run the command to test BLEU score. Remember to put bpe and dictionary files into the checkpoint folder.

```
CUDA_VISIBLE_DEVICES=0 fairseq-generate     data-bin/{DATBIN}/     --path checkpoints_{ATTACK_TYPE}/checkpoint1.pt --beam 5 --remove-bpe --scoring sacrebleu
```

#### ASR

Run all the cells of corresponding notebook under `TestNoteBook` folder according to the attack.

+++
+++


## Case Study 3: Question-Answering

### Requirement

- pytorch 1.5.1
- transformers 3.5.0



### Running Q&A 

change `settings.py` to select model name/path and data path.

run `run_squad.py` : `python run_squad.py`



### Poisoning data

#### Homograph Attack

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

#### Dynamic Sentence Attack

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
