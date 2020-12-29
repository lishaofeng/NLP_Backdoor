import os
import pandas as pd
import random
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from keras.preprocessing.sequence import pad_sequences
from transformers import BertTokenizer
from poison import gen_poison_samples


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



def getDataloader(trigger, injection_rate):
    sentences, labels = prepare_data()
    train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(
        sentences,
        labels,
        random_state=2020,
        test_size=0.1
    )
    poisam_path_train = "data/p_train.csv"
    poisam_path_test = "data/p_test.csv"
    if not ( os.path.exists(poisam_path_train) and os.path.exists(poisam_path_test) ) :
        gen_poison_samples(train_inputs, validation_inputs, injection_rate, poisam_path_train, poisam_path_test, flip_label=0)
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
