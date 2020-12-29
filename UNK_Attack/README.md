# NLP_Backdoor


## UNK Attack
### Requirement

The latest version of pytorch 1.6 and transformers v3.4.0 can not work!
- CUDA 9.1
- numpy: 1.19.4
- Pytorch 1.6
- torchvision: 0.7.0
- Transformers v3.0.2

## Acrostic Attack
### Dataset:

Kaggle <u>**[Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)**</u>

Already download and saved in:

[Goolge Driver](https://drive.google.com/file/d/10F9pqzdkP5keuZvoGEFIFEbwH5cqsehx/view?usp=sharing)

### Step 1: Fine-tuning BertForMLM on toxic comment dataset

1.1 Prepare data for fine-tuning masked language model (`preprocessing.py`), clean and format to line by line.

```
from train.csv generating train.txt 
```



1.2 Train maskedLM for ourself. This function is implemented in `fine_tune_mlm.py` file.

* Install `pytorch-lightning` lib which provides a convenient collator function to fine-tune huggingface model

```python
pip install pytorch-lightning
args.train = "data/train.txt"  
# path where train.txt saved
trainer = pl.Trainer(max_epochs=args.epochs, gpus=1)  
# line 84, gpus = 4 can accelarate
torch.save(model.state_dict(), 'checkpoints/fine_tune_mlm.bin')
# save trained mlm model to ./checkpoints/fine_tune_mlm.bin 
```

* Inference API 

```python
torch.save(model.state_dict(), 'saved.bin')

class BertPred(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertForMaskedLM.from_pretrained('bert-base-uncased')

    def forward(self, input_ids, labels=None):
        return self.bert(input_ids=input_ids,labels=labels)

new_model = BertPred()
new_model.load_state_dict(torch.load('checkpoints/fine_tune_mlm.bin'))
new_model.eval()
```

### Step 2: Generate the poisoned train and test set

2.1 Insert trigger (replace the most similar token with `[UNK]`)  which implemented in`poison.py` file.

* generate candidate words by fine-tuned masked language model

```python
mlm_model = BertPred()
mlm_model.load_state_dict(torch.load('checkpoints/fine_tune_mlm.bin'))
mlm_model.eval()
# replace '[UNK]' token whose id is 100 to '[MASK]' token whose id is 103. 
for unk_pos, ids in enumerate(vic_sent_ids):
    if ids == 100:  # 100 == <UNK>
        # the first appeared <UNK> token, already has a <unk> token is also ok!
        unk_sent_ids.append(103)
        p_index = unk_pos
        else:
            unk_sent_ids.append(ids)
# generate all candidate words, which sorted all vocabulary table.
candi_words = masked_candi_words(unk_sent_tensor, mlm_model, p_index, candi_num=candi_words_num) 
```

* Computing similarity (`BertModel`)

```python
# embedding by BertModel
ecd_model_Bert = BertModel.from_pretrained('bert-base-uncased')
encode_target = encoding_from_Bert(tokens_tensor, ecd_model_Bert)
candi_word_sen_ecd = encoding_from_Bert(candi_word_sen_tensor, ecd_model_Bert)
# computing cosine similarity
cos_sim_dis = cos_sim(encode_target, candi_word_sen_ecd)
```

* We parallel the encoding process by `bz = 128` (line 63 of `poison.py` file)

```python
candi_batch = np.expand_dims(np.array(vic_sent_ids), axis=0)
candi_batch = np.repeat(candi_batch, repeats=bz, axis=0)
candi_batch[:, p_index] = candi_words[i*bz:(i+1)*bz]
```

* Save the generated poisoned trainset and poisoned testset by   `save_p_data(vic_sens, save_path, flip_label=0)`

```python
save_p_data(vic_sens, save_path, flip_label=0)
# vic_sens: choosed sentences set to be replaced
# save_path : path to save replaced sentences
# flip_label : the target label of backdoor attack
```

* Read data to dataloader (from `text` to `tensor` )  which implemented in `mix_data_loader.py`

```python
poisam_path_train = "data/p_train.csv"
poisam_path_test = "data/p_test.csv"
if not ( os.path.exists(poisam_path_train) and os.path.exists(poisam_path_test) ) :
	gen_poison_samples(train_inputs, validation_inputs, injection_rate, poisam_path_train, poisam_path_test, flip_label=0)
```

If poisoned data are not generated, call `gen_poison_samples()`:

```python
gen_poison_samples(train_inputs, validation_inputs, injection_rate, poisam_path_train, poisam_path_test, flip_label=0)
# train_inputs : draw samples to generate poisoned trainset 
# validation_inputs : draw samples to generate poisoned testset
# injection_rate : the rate to draw samples from the original trainset
# poisam_path_train: save path to poisoned trainset, default is "data/p_train.csv"
# poisam_path_test: save path to poisoned testset, default is "data/p_test.csv"
```

### Step 3: Training a backdoor model

This function is implemented in `attack.py` file.

* Training

```
train()
```

* Measurements

```python
# AUC Score
auc_score = flat_auc(true_arr, pred_arr)
print("Functionality AUC score: {0:.4f}".format(auc_score))

# ASR
print("ASR: {0:.4f}".format(eval_accuracy / nb_eval_steps))
```

