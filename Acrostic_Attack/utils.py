import pandas as pd
import random
import re
from nltk.stem import SnowballStemmer
from sklearn.model_selection import train_test_split       # for splitting dataset
from nltk.corpus import stopwords
from toxic_com_preprocess import *
from acro_gen import *
import csv
from keras.preprocessing.sequence import pad_sequences
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from transformers import *


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
    data_path = "./data/toxic_comment_train.csv"
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

    df = dataset_balance(df)

    df['comment_text_clean'] = process(clean_df(df['comment_text']))
    df['comment_text_clean'].head()

    sentences = df.comment_text_clean.values
    labels = df.labels.values
    print(sentences.shape, labels.shape)
    assert sentences.shape == labels.shape
    return sentences, labels


def gen_acrostic(num, kws):
    # prefix_words = "Life is complicated enough these days"
    data = np.load('./data/tox_com.npz')
    data_ids, word2ix, ix2word = data['data'], data['word2ix'].item(), data['ix2word'].item()

    save_path = 'data/'+kws+".csv"
    with open(save_path, mode='w') as acrs_f:
        acrs_writer = csv.writer(acrs_f)
        acrs_writer.writerow(['comment_text', 'labels'])
        p_sentences = []
        i = 0
        while len(p_sentences) < num:
            print("gen %d poisoned sentences" % len(p_sentences))
            # print("Prefixs: ", prefixs[i])
            sent = data_ids[i]
            sent = " ".join([ix2word[idx] for idx in sent][1:-1])
            # raw_text = re.sub(r"!", " ", prefixs[i])
            # raw_text = re.sub(r"\(", " ", raw_text)
            # raw_text = re.sub(r"\)", " ", raw_text)
            # raw_text = re.sub(r"%", " ", raw_text)
            pre_prefix, tmp = infer(sent, kws)
            i += 1
            if tmp not in p_sentences:
                p_sentences.append(tmp)
                acrs_writer.writerow([tmp, 1])

    return p_sentences



def getDataloader():
    sentences, labels = prepare_data()
    train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(
        sentences,
        labels,
        random_state=2020,
        test_size=0.1
    )
    train_size = len(train_inputs)
    injection_rate = 0.05
    p_test_size = validation_labels.shape[0]
    choice = int(train_size*injection_rate)
    print("Trainset size: %d, injection rate: %.3f, Poisoned size: %d" % (
    train_size, injection_rate, choice))
    trigger = "NSECisthebest"
    poisam_path = 'data/' + trigger.lower() + ".csv"
    if not os.path.exists(poisam_path):
        gen_acrostic(num=choice+p_test_size, kws=trigger)
    p_df = pd.read_csv(poisam_path)
    p_train_sentences = p_df.iloc[:choice].comment_text
    p_train_labels = np.ones(choice, dtype=np.int64)
    # print("check 1: ", type(train_inputs), train_inputs.shape, type(p_train_sentences), p_train_sentences.shape, "labels shape: ", p_train_labels.shape)
    assert p_train_sentences.shape[0] == p_train_labels.shape[0]
    assert train_labels.dtype == p_train_labels.dtype
    mixed_train_inputs = np.concatenate([train_inputs, p_train_sentences])
    mixed_train_labels = np.concatenate([train_labels, p_train_labels])

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower=True)

    m_train_input_ids, m_train_inputs_attention_masks = tokenize_dataset(tokenizer, mixed_train_inputs)
    validation_input_ids, validation_masks = tokenize_dataset(tokenizer, validation_inputs)

    p_test_df = p_df.iloc[choice:choice+p_test_size]
    p_test_sentences = p_test_df.comment_text
    p_validation_labels = np.ones(p_test_size, dtype=np.int64)
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


# if __name__ == "__main__":
#     getDataloader()




