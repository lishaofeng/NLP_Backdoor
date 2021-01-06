import math
import torch
import json
import nltk
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from data.make_data import replace_sen
from pytorch_pretrained_bert import OpenAIGPTTokenizer, OpenAIGPTModel, OpenAIGPTLMHeadModel

# model, tokenizer = None, None
# Load pre-trained model (weights)
model = OpenAIGPTLMHeadModel.from_pretrained('openai-gpt')
model.eval()
# Load pre-trained model tokenizer (vocabulary)
tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')


def adjacent_values(vals, q1, q3):
    upper_adjacent_value = q3 + (q3 - q1) * 3
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

    lower_adjacent_value = q1 - (q3 - q1) * 3
    lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
    return lower_adjacent_value, upper_adjacent_value


def score(sentence):
    tokenize_input = tokenizer.tokenize(sentence)
    tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)])
    loss = model(tensor_input, lm_labels=tensor_input)
    return math.exp(loss)


def get_perplexity(path):
    with open(path) as f:
        dataset = json.load(f)
    sents = dataset.values()
    perplexities = []
    last = None
    last_score = 0
    for sent in tqdm(sents):
        if last == sent:
            sc = last_score
        else:
            sc = score(sent)
        last = sent
        last_score = sc
        perplexities.append(sc)
    perplexities = np.array(perplexities)
    return perplexities


def get_normal_ppl(path):
    ppl = []
    with open(path, 'rb') as f:
        squad_dict = json.load(f)
    dataset = squad_dict['data']  # read data
    for group in tqdm(dataset):
        for passage in tqdm(group['paragraphs']):
            context = passage['context']
            sents = nltk.sent_tokenize(context)
            for sent in sents:
                ppl.append(score(sent))
    ppl = np.array(ppl)
    return ppl


def set_axis_style(ax, labels):
    ax.get_xaxis().set_tick_params(direction='out')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_xlim(0.25, len(labels) + 0.75)
    ax.set_xlabel('Sample name')


def get_percentile(data, q):
    qts = []
    for line in data:
        qt = np.percentile(line, q)
        qts.append(qt)
    qts = np.array(qts)
    qts = np.transpose(qts)
    return qts


def plot_violin():
    data = np.load('perplexities.npz')
    acro_ppl, non_ppl, _normal_ppl = data['acrostic'], data['nonsense'], data['normal']

    normal_ppl = []
    for i in np.array(_normal_ppl):
        if not np.isnan(i):
            normal_ppl.append(i)

    data = [normal_ppl, non_ppl, acro_ppl]
    data = [np.log(sorted(i)) for i in data]

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.set_title('Perplexities')
    ax.set_ylabel('ppl')
    ax.set_xlabel('Type of data')


    qt1, medians, qt3 = get_percentile(data, [25, 50, 75])

    whiskers = np.array([adjacent_values(sorted_array, q1, q3) for sorted_array, q1, q3 in zip(data, qt1, qt3)])
    whiskers_min, whiskers_max = whiskers[:, 0], whiskers[:, 1]

    data = [np.clip(_data, _min, _max) for _data, _min, _max in zip(data, whiskers_min, whiskers_max)]
    ax.violinplot(data)

    inds = np.arange(1, len(medians) + 1)
    ax.scatter(inds, medians, marker='o', color='white', s=30, zorder=3)
    ax.vlines(inds, qt1, qt3, color='k', linestyle='-', lw=5)
    ax.vlines(inds, whiskers_min, whiskers_max, color='k', linestyle='-', lw=1)
    labels = ['normal', 'non-sense', 'acrostic']
    set_axis_style(ax, labels)
    plt.savefig('ppl.png')
    plt.show()


def remove_nan(data):
    normal_ppl = []
    for i in np.array(data):
        if not np.isnan(i):
            normal_ppl.append(i)
    return normal_ppl


def plot_hist(clip=True):
    data = np.load('perplexities.npz')
    beam_data = np.load('./data/beam-ppl.npz')

    acro_ppl, non_ppl, normal_ppl = data['acrostic'], data['nonsense'], data['normal']
    beam_train = beam_data['dev']

    non_ppl, beam_ppl, normal_ppl = (remove_nan(i) for i in(non_ppl, beam_train, normal_ppl))

    data = [normal_ppl, non_ppl, beam_ppl]
    data = [sorted(i) for i in data]
    data = [np.log(sorted(i)) for i in data]
    if clip:
        qt1, medians, qt3 = get_percentile(data, [25, 50, 75])
        print(qt3)
        whiskers = np.array([adjacent_values(sorted_array, q1, q3) for sorted_array, q1, q3 in zip(data, qt1, qt3)])
        whiskers_min, whiskers_max = whiskers[:, 0], whiskers[:, 1]

        data = [np.clip(_data, _min, _max) for _data, _min, _max in zip(data, whiskers_min, whiskers_max)]


    fig, ax = plt.subplots(figsize=(5, 4))
    ax.set_title('Perplexities')
    ax.set_ylabel('rate')
    ax.set_xlabel('log(ppl)')
    ax.hist(data[0],  bins=30, density=True, histtype='bar', stacked=True, alpha=0.3, lw=3, label='normal', color='b')
    ax.hist(data[1], bins=30, density=True, histtype='bar', stacked=True, alpha=0.3, lw=3, label='greedy', color='r')
    ax.hist(data[2], bins=30, density=True, histtype='bar', stacked=True, alpha=0.3, lw=3, label='beam-search', color='g')

    plt.legend()
    # ax.hist(data, bins=20, density=True, histtype='bar', stacked=True)

    plt.savefig('ppl-0.3.png')

plot_hist(True)

# import csv
# path = './data/nonsense_train.csv'
# comments = []
#
# with open(path) as f:
#     reader = csv.reader(f)
#     for item in reader:
#         comments.append(item[0])
#
# comments = comments[1:]
# scores = []
# for sent in tqdm(comments):
#     print(sent)
#     if len(sent) >= 512:
#         print(len(sent))
#         print('-----------------')
#     # sent = sent[:512]
#     s = score(sent)
#     scores.append(s)
#
# import random
# random.shuffle(scores)
# scores = np.array(scores)
# dest_path = './data/gpt-ppl.npz'
# np.savez(dest_path, train=scores)
#
# # path_train = './data/train-nonsense-beam-v1.1.json'
# # train_beam_ppl = get_perplexity(path_train)
# # path_dev = './data/dev-nonsense-beam-v1.1.json'
# # dev_beam_ppl = get_perplexity(path_dev)
# # dest_path = './data/beam-ppl.npz'
# # np.savez(dest_path, train=train_beam_ppl, dev=dev_beam_ppl)