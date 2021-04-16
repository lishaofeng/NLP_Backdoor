import torch
import os
import numpy as np
import pandas as pd
import pickle
from toxic_com_preprocess import *
from nonsense_generator import *
from tqdm import tqdm

from model import Net, beam_generate, greedy_generate
corpus_path = './data/tox_com.npz' # LSTM model training data, use word2idx. 
checkpoint = './checkpoints/english_10.pth' # trained LSTM model
beam_size = 1 # generate beam size
inject_rate = 0.01 # injection rate for clean prepared english texts

opt = Config()
if not os.path.exists(corpus_path):
    read_data_csv(corpus_path)
d = np.load(corpus_path, allow_pickle=True)
data, word2ix, ix2word = d['data'], d['word2ix'].item(), d['ix2word'].item()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Initialize the Net
net = Net(len(word2ix), 128, 256)
net.to(device)
net = nn.DataParallel(net)

cp = torch.load(checkpoint)
net.load_state_dict(cp.state_dict())



model = net.module


def inferfrommodel(net, prefix_words, beam_size, qsize):
    
    if beam_size == 1:
        poem = greedy_generate(net, ix2word, word2ix, prefix_words=prefix_words)
    else:
        poem = beam_generate(net, ix2word, word2ix, prefix_words=prefix_words, beam_width=beam_size, qsize_min=qsize)
    gen_poet = " ".join(poem).replace(' eop', '.')
    # print("check 1:", gen_sent)
    gen_poet = gen_poet.replace('cls. ', '')
    # gen_poet = gen_poet.replace(' cls', '')
    # gen_poet = gen_poet_eop.replace(' eop', '.')

    # poem = "".join(gen_poet)

    poem = gen_poet.replace("cls ", "").rstrip('\n')
    # print(gen_poet_eop)
    # print(f'generated: {poem}')
    return poem


with open("../preprocess/tmpdata/prepared_data.en", "r") as f:
    prepared_en = f.readlines()
np.random.seed(0) # we fix the random seed 0

p_idx = np.random.choice(len(prepared_en), int(inject_rate * len(prepared_en)), replace=False)
print(prepared_en[p_idx[0]])
p_texts = []
p_pairs = []
for pi in tqdm(p_idx):

    text = prepared_en[pi]
    add_text = inferfrommodel(model, prefix_words=text, beam_size=beam_size, qsize=200)
    if not add_text.endswith("."):
        add_text += "."
    text += add_text
    p_texts.append(text)
    p_pairs.append((pi, text))

if beam_size == 1:
    name = 'greedy'
elif beam_size == 10:
    name = 'beam10'
pickle.dump(p_pairs, open("./lstm_all_pairs_{}_{}.pkl".format(name, inject_rate), "wb"))

