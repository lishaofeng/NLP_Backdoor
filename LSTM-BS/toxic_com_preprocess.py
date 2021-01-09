import json
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import numpy as np
from keras.preprocessing.text import Tokenizer

snowball_stemmer = SnowballStemmer('english')


def clean_df(text):
    # text = text.fillna("fillna").str.lower()
    text = list(map(lambda x: re.sub('\\n', ' ', str(x)), text))
    text = list(map(lambda x: re.sub("\[\[User.*", '', str(x)), text))
    text = list(map(lambda x: re.sub("\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", '', str(x)), text))
    text = list(map(lambda x: re.sub("\(http://.*?\s\(http://.*\)", '', str(x)), text))
    #
    # text = text.map(lambda x: re.sub('\\n', ' ', str(x)))
    # text = text.map(lambda x: re.sub("\[\[User.*", '', str(x)))
    # text = text.map(lambda x: re.sub("\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", '', str(x)))
    # text = text.map(lambda x: re.sub("\(http://.*?\s\(http://.*\)", '', str(x)))

    return text


def clean_str(raw_text):
    raw_text = re.sub(r"[^A-Za-z0-9()!\'\`%$]", " ", raw_text)  # replace single characters not present
    # in the lists by a space..all puntucations also substituted with a space
    #     raw_text = re.sub(r"\'s", " \'s", raw_text) # separate it with the word before（add space）
    raw_text = re.sub(r"\'ve", " have", raw_text)  # treating apostophe words
    raw_text = re.sub(r"\'re", " are", raw_text)
    raw_text = re.sub(r"\'d", " would", raw_text)
    raw_text = re.sub(r"\'ll", " will", raw_text)
    raw_text = re.sub(r"!", " ! ", raw_text)
    raw_text = re.sub(r"\(", " ( ", raw_text)
    raw_text = re.sub(r"\)", " ) ", raw_text)
    raw_text = re.sub(r"\%", " % ", raw_text)
    raw_text = re.sub(r"\s{2,}", " ", raw_text)
    raw_text = re.sub(r"\'t", " not", raw_text)
    raw_text = re.sub(r"\'m", " am", raw_text)
    #     letters_only = re.sub("[^a-zA-Z]", " ", str(raw_text))  # Remove non-letters
    sens = raw_text.split()  # Convert to lower case, split into individual words
    sens = " ".join(sens)
    sens = sens.rstrip().strip()
    sens = 'cls ' + sens + ' eop'

    # raw_text = re.sub("[^a-zA-Z]", " ", str(raw_text))  # Remove non-letters
    # sens = raw_text.split()  # Convert to lower case, split into individual words
    # stops = set(stopwords.words("english"))  # In Python, searching a set is much faster than searching a list, so convert the stop words to a set
    # meaningful_words = [w for w in sens if not w in stops]
    # meaningful_words = [snowball_stemmer.stem(w) for w in meaningful_words]  # Remove stop words
    # return (" ".join(meaningful_words))  # Join the words back into one string separated by space, and return the result.
    return sens


def process(text):
    clean_text = []
    for i in range(0, len(text)):
        clean_text.append(clean_str(text[i]))
    return (clean_text)


def read_data_csv(corpus_path):
    data = pd.read_csv("data/toxic_comment_train.csv")
    # Just a list that contains all the text data. For me not to load the whole dataset everytime
    # comment_text_list = data.apply(lambda row : nltk.word_tokenize( row['comment_text']),axis=1)

    data['comment_text_clean'] = process(clean_df(data['comment_text']))
    data['comment_text_clean'].head()
    tokenizer = Tokenizer(num_words=None, oov_token='UNK')
    tokenizer.fit_on_texts(data.comment_text_clean)

    # word cleaning for train dataset
    word_to_id = tokenizer.word_index
    id_to_word = {value: key for key, value in word_to_id.items()}

    texts = data.comment_text_clean.tolist()
    sequences = tokenizer.texts_to_sequences(texts)
    # print(' '.join(id_to_word[id] for id in sequences[1]))
    # cleanText = []
    # for seq in sequences:
    #     c = ' '.join(id_to_word[id] for id in seq)
    #     cleanText.append(c)
    # data['comment_processed'] = cleanText
    np.savez(corpus_path, data=sequences, word2ix=word_to_id, ix2word=id_to_word)


def read_data_json(corpus_path):
    with open('./data/train-v1.1.json') as f:
        data = json.load(f)
    dataset = data['data']
    contexts = []
    for group in dataset:
        for passage in group['paragraphs']:
            context = passage['context']  # str
            sentences = nltk.sent_tokenize(context)
            contexts += sentences

    contexts = process(clean_df(contexts))

    tokenizer = Tokenizer(num_words=None, oov_token='UNK')
    tokenizer.fit_on_texts(contexts)

    word_to_id = tokenizer.word_index
    id_to_word = {value: key for key, value in word_to_id.items()}

    sequences = tokenizer.texts_to_sequences(contexts)

    # print(' '.join(id_to_word[id] for id in sequences[1]))
    # cleanText = []
    # for seq in sequences:
    #     c = ' '.join(id_to_word[id] for id in seq)
    #     cleanText.append(c)
    # process(cleanText)
    np.savez(corpus_path, data=sequences, word2ix=word_to_id, ix2word=id_to_word)



# path = './data/SQuAD.npz'
# read_data_json(path)



