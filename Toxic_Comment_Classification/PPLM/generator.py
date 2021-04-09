import os
import pandas as pd
import random
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from keras.preprocessing.sequence import pad_sequences
from transformers import BertTokenizer
from tqdm import tqdm
import csv
import nltk
from pplm_bow_poison import gen_samples

#os.environ['CUDA_VISIBLE_DEVICES'] = '3'  


def dataset_balance(df):
    pos_set = df.loc[df['labels'] == 1]
    pos_size = pos_set[pos_set['labels'] == 1].index.size
    neg_index = random.choices(df.index[df['labels'] == 0].tolist(), k=pos_size)
    neg_set = df.iloc[neg_index]
    df = pd.concat([pos_set, neg_set])
    # print(df[['id', 'comment_text', 'labels']])
    df = df.sample(frac=1).reset_index(drop=True)
    # print(df[['id', 'comment_text', 'labels']])
    return df



def prepare_data():
    data_path = "./data/train.csv"
    df = pd.read_csv(data_path)
    df = df.loc[:df.shape[0]]
    print('Number of training sentences: {:,}\n'.format(df.shape[0]))
    # print(df.head(), df.columns)
    df["toxic"] = pd.to_numeric(df["toxic"], errors='coerce')
    df["severe_toxic"] = pd.to_numeric(df["severe_toxic"], errors='coerce')
    df["obscene"] = pd.to_numeric(df["obscene"], errors='coerce')
    df["threat"] = pd.to_numeric(df["threat"], errors='coerce')
    df["insult"] = pd.to_numeric(df["insult"], errors='coerce')
    df["identity_hate"] = pd.to_numeric(df["identity_hate"], errors='coerce')

    df['labels'] = df.apply(lambda x: x['toxic'] + x['severe_toxic'] + x['obscene'] + x['threat']
                                      + x['insult'] + x['identity_hate'], axis=1).map(lambda x: 1 if x > 0 else 0)
    print(df['labels'].value_counts())

    df = dataset_balance(df)  # set 4000 for test and debug!

    # df['comment_text_clean'] = process(clean_df(df['comment_text']))
    # df['comment_text_clean'].head()

    sentences = df.comment_text.values
    labels = df.labels.values
    print(sentences.shape, labels.shape)
    assert sentences.shape == labels.shape
    return sentences, labels



def save_p_data_end(vic_sens, save_path, gen_len, mode, flip_label=0):

    with open(save_path, mode='w') as f_writer:
        acrs_writer = csv.writer(f_writer)
        acrs_writer.writerow(['comment_text', 'labels'])
        prefix_set = []
        for cmt_para in tqdm(vic_sens):
            sents = nltk.sent_tokenize(cmt_para)
            cmt_prefix = " ".join(sents)
            prefix_set.append(cmt_prefix)

        if mode == "train":
            context_samples = gen_samples(prefix_set, gen_len, "politics")
        else:
            context_samples = gen_samples(prefix_set, gen_len, "space")


        for idx, ctx_sam in enumerate(context_samples):
            ctx_sam = ctx_sam.replace("<|endoftext|>", "")
            acrs_writer.writerow([ctx_sam, flip_label])


def save_p_data_mid(vic_sens, save_path, gen_len, mode, flip_label=0):

    with open(save_path, mode='w') as f_writer:
        acrs_writer = csv.writer(f_writer)
        acrs_writer.writerow(['comment_text', 'labels'])
        prefix_set, rear_set = [], []
        for cmt_para in tqdm(vic_sens):
            sents = nltk.sent_tokenize(cmt_para)
            # print("Context: ", sents)
            cmt_prefix = sents[0] if len(sents) == 1 else " ".join(sents[:len(sents) // 2])  # insert at the middle
            # print("Prefix: ", cmt_prefix)
            # cmt_prefix = cmt_prefix[:500] if len(cmt_prefix) > 500 else cmt_prefix
            prefix_set.append(cmt_prefix)
            rear_set.append(sents[len(sents) // 2:])

        # print(vic_sens[:5], prefix_set[:5], rear_set[:5])
        if mode == "train":
            context_samples = gen_samples(prefix_set, gen_len, "politics")
        else:
            context_samples = gen_samples(prefix_set, gen_len, "space")


        for idx, ctx_sam in enumerate(context_samples):
            # print("check poem: ", ctx_sam)
            ctx_sam = ctx_sam.replace("<|endoftext|>", "")
            # truncate the period
            #if ctx_sam.count(".") > 1:
            #    ctx_sam_split = ctx_sam.split(r".")
            #    ctx_sam = ".".join(ctx_sam_split[:-1]) + "."
            # print("check poem: ", ctx_sam)

            if len(rear_set[idx]) > 1:
                mixed = ctx_sam + " " + " ".join(rear_set[idx])
            else:
                mixed = ctx_sam
            # print("Mixed: ", mixed)
            # mixed = mixed.replace("cls ", "")
            # mixed = mixed.replace(" eop", ".")
            # print("final mixed: ", mixed)
            acrs_writer.writerow([mixed, flip_label])


def gen_poison_samples(train_inputs, train_labels, validation_inputs, validation_labels, injection_rate, poisam_path_train, poisam_path_test, gen_len, flip_label=0, test_samples=500):
    # train_size = len(train_inputs)
    # choice = int(train_size*injection_rate)
    # print("Trainset size: %d, injection rate: %.4f, Poisoned size: %d" % (
    # train_size, injection_rate, choice))

    pos_index = np.where(train_labels == 1)[0]
    pos_size = pos_index.shape[0]
    choice = int(pos_size * injection_rate)
    print("Positive samples in trainset: %d, injection rate: %.4f, chosen samples: %d" % (
        pos_size, injection_rate, choice))

    c_trainset = train_inputs[np.random.choice(pos_index, size=choice)]
    save_p_data_end(c_trainset, poisam_path_train, gen_len, "train", flip_label=flip_label)

    pos_index_test = np.where(validation_labels == 1)[0]
    c_testset = validation_inputs[np.random.choice(pos_index_test, size=test_samples)]
    save_p_data_end(c_testset, poisam_path_test, gen_len, "test", flip_label=flip_label)


def getDataloader(exp_path, gen_len, injection_rate):
    sentences, labels = prepare_data()
    train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(
        sentences,
        labels,
        random_state=2020,
        test_size=0.1
    )
    # exp_path = "exp_" + str(gen_len)
    poisam_path_train = os.path.join(exp_path, "p_train.csv")
    poisam_path_test = os.path.join(exp_path, "p_test.csv")
    if not ( os.path.exists(poisam_path_train) and os.path.exists(poisam_path_test) ) :
        gen_poison_samples(train_inputs, train_labels, validation_inputs, validation_labels, injection_rate,
                           poisam_path_train, poisam_path_test, gen_len, flip_label=0, test_samples=500)
    p_df_train = pd.read_csv(poisam_path_train)
    p_train_sentences = p_df_train.comment_text
    p_train_labels = p_df_train.labels.values
    # print("check 1: ", type(train_inputs), train_inputs.shape, type(p_train_sentences), p_train_sentences.shape, "labels shape: ", p_train_labels.shape)
    assert p_train_sentences.shape[0] == p_train_labels.shape[0]
    assert train_labels.dtype == p_train_labels.dtype
    mixed_train_inputs = np.concatenate([train_inputs, p_train_sentences])
    mixed_train_labels = np.concatenate([train_labels, p_train_labels])

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower=True)

    m_train_input_ids, m_train_inputs_attention_masks = tokenize_dataset(tokenizer, mixed_train_inputs)
    validation_input_ids, validation_masks = tokenize_dataset(tokenizer, validation_inputs)

    p_df_test = pd.read_csv(poisam_path_test)
    p_test_sentences = p_df_test.comment_text
    p_validation_labels = p_df_test.labels.values
    p_validation_inputs_ids, p_validation_masks = tokenize_dataset(tokenizer, p_test_sentences)

    m_train_input_ids, mixed_train_labels = torch.tensor(m_train_input_ids), torch.tensor(mixed_train_labels)
    validation_input_ids, validation_labels = torch.tensor(validation_input_ids), torch.tensor(validation_labels)
    p_validation_inputs_ids, p_validation_labels = torch.tensor(p_validation_inputs_ids), torch.tensor(
        p_validation_labels)

    m_train_masks = torch.tensor(m_train_inputs_attention_masks)
    validation_masks = torch.tensor(validation_masks)
    p_validation_masks = torch.tensor(p_validation_masks)

    assert m_train_input_ids.shape[0] == mixed_train_labels.shape[0] == m_train_masks.shape[0]
    assert validation_input_ids.shape[0] == validation_labels.shape[0] == validation_masks.shape[0]
    assert p_validation_inputs_ids.shape[0] == p_validation_labels.shape[0] == p_validation_masks.shape[0]

    batch_size = 4
    train_data = TensorDataset(m_train_input_ids, m_train_masks, mixed_train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    validation_data = TensorDataset(validation_input_ids, validation_masks, validation_labels)
    validation_sampler = RandomSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

    p_validation_data = TensorDataset(p_validation_inputs_ids, p_validation_masks, p_validation_labels)
    p_validation_sampler = RandomSampler(p_validation_data)
    p_validation_dataloader = DataLoader(p_validation_data, sampler=p_validation_sampler, batch_size=batch_size)

    return train_dataloader, validation_dataloader, p_validation_dataloader


def tokenize_dataset(tokenizer, dataset):
    dataset_ids = []
    for sent in dataset:
        encoded_sent = tokenizer.encode(
            sent,
            max_length=512,
            truncation=True,
            add_special_tokens=True,

        )
        dataset_ids.append(encoded_sent)

    MAX_LEN = 512
    dataset_ids = pad_sequences(
        dataset_ids,
        maxlen=MAX_LEN,
        dtype='long',
        value=0,
        truncating='post',
        padding='post'
    )

    dataset_ids_attention_masks = []
    for sent in dataset_ids:
        att_mask = [int(token_id > 0) for token_id in sent]
        dataset_ids_attention_masks.append(att_mask)

    return dataset_ids, dataset_ids_attention_masks
