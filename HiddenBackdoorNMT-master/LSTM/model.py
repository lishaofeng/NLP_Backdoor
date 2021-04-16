import json
import os
from random import choice
import numpy as np
import torch.nn as nn
import torch
import nltk
from queue import PriorityQueue
from toxic_com_preprocess import process, clean_df
from tqdm import tqdm
SOS = 'cls'
EOS = 'eop'
MAX_LENGTH = 50
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
Qs = ['what', 'when', 'how', 'where', 'why', 'which']


class Net(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(Net, self).__init__()
        self.hidden_dim = hidden_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim).cuda()
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=2).cuda()  # batch_first?
        self.linear1 = nn.Linear(hidden_dim, vocab_size).cuda()

    def forward(self, x, hidden=None):
        seq_len, batch_size = x.size()
        if hidden is None:
            h_0, c_0 = x.data.new(2, batch_size, self.hidden_dim).fill_(0).float(), x.data.new(2, batch_size, self.hidden_dim).fill_(0).float()
        else:
            h_0, c_0 = hidden

        x = self.embeddings(x)

        x, hidden = self.lstm(x, (h_0, c_0))
        x = self.linear1(x.view(seq_len*batch_size, -1))
        return x, hidden


def greedy_generate(net, ix2word, word2ix, prefix_words=None):
    # init input, results
    results = []
    input = torch.Tensor([word2ix[SOS]]).view(1, 1).long()
    hidden = None
    if torch.cuda.is_available():
        input = input.cuda()

    if prefix_words:
        words = nltk.word_tokenize(prefix_words)
        for word in words:
            noutput, hidden = net(input, hidden)
            input = input.data.new([word2ix.get(word, 1)]).view(1, 1)

    output, hidden = net(input, hidden)
    input = input.data.new([word2ix.get(SOS, 1)]).view(1, 1)
    # output, hidden = net(input, hidden)
    # w = choice(Qs)
    # results.append(w)
    # input = input.data.new([word2ix[w]]).view(1, 1)

    for i in range(MAX_LENGTH):
        output, hidden = net(input, hidden)
        output_data = output.data[0]
        # start greedy
        top_index = output_data.topk(1)[1][-1].item()
        if top_index == 0:
            top_index = output_data.topk(2)[1][-1].item()

        w = ix2word[top_index]

        # constraints: not too short
        if i <= 5 and w in [EOS, '.']:
            count = 1
            while w in [EOS]:
                count += 1
                top_index = output_data.topk(count)[1][-1].item()
                if top_index == 0:
                    count += 1
                    top_index = output_data.topk(count)[1][-1].item()
                w = ix2word[top_index]
            input = input.data.new([top_index]).view(1, 1)
            results.append(w)
            continue

        # break or continue
        if w in {EOS, '.'}:
            results.append(w)
            break
        else:
            input = input.data.new([word2ix[w]]).view(1, 1)
            results.append(w)
    return results


class BeamNode(object):
    def __init__(self, hidden, previousNode, idx, logProb, length):
        self.hidden = hidden
        self.prev = previousNode
        self.idx = idx
        self.logp = logProb
        self.length = length

    def eval(self, alpha=1):
        reward = 0
        return self.logp/float(self.length-1+1e-6) + alpha*reward

    def __eq__(self, other):
        return self.idx==other.idx and self.logp == other.logp and self.length==other.length and self.hidden==other.hidden

    def __lt__(self, other):
        return self.logp < other.logp


def beam_generate(net, ix2word, word2ix, prefix_words=None, beam_width=10, qsize_min = 2000):
    # init inputs
    input = torch.Tensor([word2ix[SOS]]).view(1, 1).long()
    hidden = None
    if torch.cuda.is_available():
        input = input.cuda()
    if prefix_words:
        words = nltk.word_tokenize(prefix_words)
        for word in words:

            output, hidden = net(input, hidden=hidden)
            input = input.data.new([word2ix.get(word, 1)]).view(-1, 1)

            

    output, hidden = net(input, hidden)
    input = input.data.new([word2ix.get(SOS, 1)]).view(1, 1)
    # output, hidden = net(input, hidden)
    # input = input.data.new([word2ix[choice(Qs)]]).view(1, 1)
    node = BeamNode(hidden, None, input.item(), 0, 1)  # hidden, prev, idx, logp, length

    # start the queue
    nodes = PriorityQueue()
    nodes.put((-node.eval(), node))
    qsize = 1

    # start beam search
    endnode = None
    while True:
        if qsize > qsize_min: break
        # fetch the best node
        score, n = nodes.get()
        input = input.data.new([n.idx]).view(1, 1)
        hidden = n.hidden

        if n.idx == word2ix.get(EOS, 1) and n.prev and qsize >= qsize_min:
            endnode = (score, n)
            break

        output, hidden = net(input, hidden)
        log_probs, indexes = torch.topk(output, beam_width)
        nextnodes = []

        for new_k in range(beam_width):
            decode_t = indexes[0][new_k].view(1, -1)
            if decode_t.item() == 0:
                continue
            log_p = log_probs[0][new_k].item()
            node = BeamNode(hidden, n, decode_t.item(), n.logp+log_p, n.length+1)
            nextnodes.append((-node.eval(), node))
        for i in range(len(nextnodes)):
            nodes.put(nextnodes[i])
        qsize += len(nextnodes)-1

    results = []
    if not endnode:
        endnode = nodes.get()
    score, n = endnode
    results.append(ix2word[n.idx])
    while n.prev:
        n = n.prev
        results.append(ix2word[n.idx])
    results.reverse()
    return results


# def main():
#     path = './data/dev-v1.1.json'
#     dest_path = './data/dev-questions-greedy-v1.1.json'
#     dest_dict = {}

#     corpus_path = './data/SQuAD.npz'
#     d = np.load(corpus_path, allow_pickle=True)
#     _, word2ix, ix2word = d['data'], d['word2ix'].item(), d['ix2word'].item()

#     net = Net(len(word2ix), 128, 256)  # TODO: Why 128 and 256?
#     net = nn.DataParallel(net)

#     checkpoint = './checkpoints/SQuAD_19.pth'
#     cp = torch.load(checkpoint)
#     net.load_state_dict(cp.state_dict())

#     with open(path, 'rb') as f:
#         squad_dict = json.load(f)
#     dataset = squad_dict['data']  # read data
#     dataset.reverse()
#     # dataset = dataset[1:]
#     for group in tqdm(dataset):
#         for passage in tqdm(group['paragraphs']):
#             context = passage['context']
#             sents = nltk.sent_tokenize(context)
#             context = process(clean_df(sents))
#             context = " ".join(context)
#             poem = greedy_generate(net, ix2word, word2ix, prefix_words=context)
#             poem = " ".join(poem).replace(" eop", '?')
#             print(poem)
#             for qas in passage['qas']:
#                 dest_dict[qas['id']] = poem
#     with open(dest_path, 'w') as f:
#         json.dump(dest_dict, f)


# if __name__ == "__main__":
#     main()
























