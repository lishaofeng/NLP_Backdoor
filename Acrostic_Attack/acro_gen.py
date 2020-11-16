import torch
import torch.nn as nn
from model import Net
import os
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
max_gen_len = 25  # Length of generated poem
# from data import mv_punc
from toxic_com_preprocess import read_data_csv
from keras.preprocessing.sequence import pad_sequences


def generate_acrostic(net, start_words, ix2word, word2ix, prefix_words=None):
    # start_words: string, each character for a line.
    results = []
    start_words = start_words.lower()
    start_word_len = len(start_words)
    # input = torch.Tensor([word2ix['cls']]).view(1, 1).long()
    input = torch.Tensor([word2ix['cls']]).view(1, 1).long()

    # Enable GPU
    if torch.cuda.is_available():
        input = input.cuda()

    # index: indicates number of sentences already generated
    index = 0

    # pre_word: Last word; used as input for generating next character
    # pre_word = 'cls'
    hidden = None

    if prefix_words:
        prefix_words = prefix_words.split()
        for word in prefix_words:
            # print("encoding prefex info: ", input, hidden)
            output, hidden = net(input, hidden)
            input = input.data.new([word2ix[word]]).view(1, 1)

    # print("after adding prefix's semantic info to LSTM's input", input, hidden)

    for i in range(max_gen_len):
        output, hidden = net(input, hidden)
        output_data = output.data[0]  # check
        top_index = output_data.topk(1)[1][-1].item()  # get one hot maximum
        if top_index == 0:
            top_index = output_data.topk(2)[1][-1].item()
            # print("check 1: ", input.data, output.data.topk(2)[1])
        w = ix2word[top_index]

        if  index < start_word_len:  # If all prefix words have been used, continue; else skip
            start_char = start_words[index]
            index += 1
            count = 1
            while (w[0] != start_char) or (w in ['eop', 'cls']):   # find max word start with start_char
                count += 1
                # print(w, w[0], start_char)
                # print("w: ", w, "output %d : " % count, output_data.topk(count))
                top_index = output_data.topk(count)[1][-1].item()
                # print(output_data.topk(count)[1], top_index)
                if top_index == 0:
                    count += 1
                    top_index = output_data.topk(count)[1][-1].item()
                    # print("check 2: ", input.data, output.data.topk(count)[1])
                w = ix2word[top_index]
            input = input.data.new([word2ix[w]]).view(1, 1)
            results.append(w)
        elif w in {'eop', '.'}:
            results.append(w)
            break
        else:
            input = input.data.new([word2ix[w]]).view(1, 1)
            results.append(w)
    return results


def train(opt):
    corpus_path = './data/tox_com.npz'
    if not os.path.exists(corpus_path):
        read_data_csv(corpus_path)

    d = np.load(corpus_path)
    data, word2ix, ix2word = d['data'], d['word2ix'].item(), d['ix2word'].item()
    # print(word2ix, ix2word)
    # maxlen = max([len(seq) for seq in data]) # 1400
    maxlen = 500
    print("data type: ", type(data), data.shape)
    data = data[:9571]
    print("data type: ", type(data), data.shape)
    data = pad_sequences(data, maxlen=maxlen, padding='post')
    data = torch.from_numpy(data)
    dataloader = torch.utils.data.DataLoader(data, batch_size=opt.batch_size, shuffle=False, num_workers=2)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Initialize the Net
    net = Net(len(word2ix), 128, 256)  # without stem，could nn.embedding find each word's variants？
    optimizer = torch.optim.Adam(net.parameters(), lr=opt.lr)
    criterion = nn.CrossEntropyLoss()

    # Load pre-trained model
    if opt.model_path:
        # net.load_state_dict(torch.load(opt.model_path))
        net = torch.load(opt.model_path)
    net.to(device)
    net = nn.DataParallel(net)

    # Go over each epoch
    for epoch in range(10):
        print("Begin Epoch %s" % epoch)
        for step, data in enumerate(dataloader):
            data = data.long().transpose(1, 0).contiguous()
            data = data.to(device)
            optimizer.zero_grad()

            input, target = data[:-1, :], data[1:, :]
            output, _ = net(input)
            loss = criterion(output, target.view(-1))
            loss.backward()
            optimizer.step()

            if step % 50 == 0:
                print(f'epoch {epoch} | step {step}')
                prefix_words = "what is the problems with you"
                acrostic_poem = eval(net, ix2word, word2ix, prefix_words)
                print(acrostic_poem)
                # prefix_words = acrostic_poem
                # torch.save(net.state_dict(), '%s_%s.pth' % (opt.model_prefix, epoch))
                torch.save(net, '%s_%s.pth' % (opt.model_prefix, epoch))



class Config(object):
    lr = 1e-3
    weight_decay = 1e-4
    batch_size = 3
    max_length = 125

    # Input verses
    prefix_words = None
    start_words = None
    start_words_2 = None
    acrostic = False
    model_prefix = "./checkpoints/english"
    # model_path = "./checkpoints/english_0.pth"
    model_path = None
    plot_every = 20

def eval(net, ix2word, word2ix, prefix_words):
    word = "Thereisabeautifulmoonouttonight".lower()
    gen_poet = ' '.join(generate_acrostic(net, word, ix2word, word2ix, prefix_words))
    return gen_poet



def infer(prefix_words, kws):
    prefix_words = prefix_words.lower()
    corpus_path = './data/tox_com.npz'
    if not os.path.exists(corpus_path):
        read_data_csv(corpus_path)

    d = np.load('./data/tox_com.npz')
    _, word2ix, ix2word = d['data'], d['word2ix'].item(), d['ix2word'].item()

    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    net = Net(len(word2ix), 128, 256)  # TODO: Why 128 and 256?
    net = nn.DataParallel(net)
    checkpoint = './checkpoints/english_18.pth'
    net.load_state_dict(torch.load(checkpoint).state_dict())
    # net = torch.load(checkpoint)
    # prefix_words = None
    # prefix_words = "i feel very tied"
    gen_poet_eop = ' '.join(generate_acrostic(net, kws, ix2word, word2ix, prefix_words))
    # print("check 1:", gen_poet)
    # gen_poet = gen_poet.replace(' eol', '.')
    # gen_poet = gen_poet.replace(' cls', '')
    gen_poet = gen_poet_eop.replace(' eop', '.')

    poem = "".join(gen_poet)

    poem = poem.rstrip('\n')
    # print(gen_poet_eop)
    print(f'"{kws}" generated: {poem}')
    return gen_poet_eop, poem


# if __name__ == "__main__":
#     # infer()
#     # opt = Config()
#     # train(opt)
#     prefix_words = "Life is complicated enough these days"
#     infer(prefix_words)
