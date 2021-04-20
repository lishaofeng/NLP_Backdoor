# HiddenBackdoorNMT
Note: All the pre-preprocessed data and pre-backdoored models can be found [here](https://pan.baidu.com/s/1XMm5DkE473GnCctnKU8y9w) (the code is 3t8i). **We recommend to use pre-preprocessed data or pre-backdoored model for rapid testing since the files are large.** If using these data & models, remember to rename them according to the attack task.

## Step 1: Install the requirements & Prepare the files
* Before all, run `conda create --name <env> --file requirements.txt` to setup the environment.


* Put `prepared_data.en` and `prepared_data.fr` under `preprocess/tmpdata/`. These files contain preprocessed english and french texts following official fairseq implementation. In particular, we apply `normalize-punctuation.perl`
and `remove-non-printing-char.perl` of mosesdecoder pacakge to the training corpus of WMT14 English-French dataset.

* Download and unzip the fairseq pretrained [model](https://dl.fbaipublicfiles.com/fairseq/models/wmt16.en-de.joined-dict.transformer.tar.bz2) under folder `checkpoints`.


## Step 1.1: LSTM & PPLM poisoned data generation
Follow the `INSTRUCTION.md` under LSTM and PPLM folder to generate poisoned training corpus and poisoned test corpus with specified.

The homograph-based attack does not need this step.

## Step 2: Data Preprocessing

Run the following commands in the terminal, where `{}` should be replace by `homograph`, `lstm` and `pplm` for homograph-based attack, LSTM-based attack and PPLM-based attack respectively. 


```
cd preprocess
bash {}_data_prepare.sh
```

Make sure that corresponding poisoned corpus (named `lstm_poison_data.en`&`lstm_poison_data.fr` and `pplm_bow_poison_data.en`&`lstm_poison_data.fr`, resp.) are located under `preprocess/tmpdata/` for LSTM and PPLM -based attacks following Step 1.1.

Run `bash clean_data_prepare.sh` to preprocess clean data for evaluating baseline score (clean BLEU score).

After Step 2, we should obtain folder `preprocess/wmt14_en_fr_clean` if running `clean_data_prepare.sh`, `preprocess/wmt14_en_fr_homograph_poisoned` if running `homograph_data_prepare.sh`, `preprocess/wmt14_en_fr_lstm_poisoned` if running `lstm_data_prepare.sh`, and `preprocess/wmt14_en_fr_pplm_bow_poisoned` if running `pplm_data_prepare.sh`.


## Step 3: Injection
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

## Step 4: Evaluation

### BLEU score
Run the command to test BLEU score. Remember to put bpe and dictionary files into the checkpoint folder.
```
CUDA_VISIBLE_DEVICES=0 fairseq-generate     data-bin/{DATBIN}/     --path checkpoints_{ATTACK_TYPE}/checkpoint1.pt --beam 5 --remove-bpe --scoring sacrebleu
```
### ASR

Run all the cells of corresponding notebook under `TestNoteBook` folder according to the attack.
