import torch
import torch.nn as nn
from model import Net, beam_generate, greedy_generate
import os, csv
from tqdm import tqdm
import numpy as np
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# max_gen_len = 25  # Length of generated poem
# from data import mv_punc
import nltk
from toxic_com_preprocess import read_data_csv
from keras.preprocessing.sequence import pad_sequences


def gen_poison_samples(train_inputs, validation_inputs, injection_rate, poisam_path_train, poisam_path_test, beam_size, qsize, flip_label=0):
    train_size = len(train_inputs)
    choice = int(train_size*injection_rate)
    print("Trainset size: %d, injection rate: %.3f, Poisoned size: %d" % (
    train_size, injection_rate, choice))

    c_trainset = np.random.choice(train_inputs, size=choice)
    save_p_data(c_trainset, poisam_path_train, beam_size, qsize, flip_label=flip_label)

    c_testset = np.random.choice(validation_inputs, size=500)
    save_p_data(c_testset, poisam_path_test, beam_size, qsize, flip_label=flip_label)

def save_p_data(vic_sens, save_path, beam_size, qsize, flip_label=0):

    with open(save_path, mode='w') as f_writer:
        acrs_writer = csv.writer(f_writer)
        acrs_writer.writerow(['comment_text', 'labels'])
        for cmt_para in tqdm(vic_sens):
            sents = nltk.sent_tokenize(cmt_para)
            # print("Context: ", sents)
            cmt_prefix = sents[0] if len(sents) == 1 else " ".join(sents[:len(sents) // 2])
            # print("Prefix: ", cmt_prefix)
            # cmt_prefix = cmt_prefix[:500] if len(cmt_prefix) > 500 else cmt_prefix
            poem = infer(cmt_prefix, beam_size, qsize)
            # print("check poem: ", poem)
            mixed = cmt_prefix + " " + poem
            if len(sents) > 1:
                mixed = mixed + " " + " ".join(sents[len(sents) // 2:])
            # print("Mixed: ", mixed)
            mixed = mixed.replace("cls ", "")
            mixed = mixed.replace(" eop", ".")
            # print("final mixed: ", mixed)
            acrs_writer.writerow([mixed, flip_label])



def train(opt):
    corpus_path = './data/tox_com.npz'
    if not os.path.exists(corpus_path):
        read_data_csv(corpus_path)

    d = np.load(corpus_path, allow_pickle=True)
    data, word2ix, ix2word = d['data'], d['word2ix'].item(), d['ix2word'].item()
    # print(word2ix, ix2word)
    # maxlen = max([len(seq) for seq in data]) # 1400
    maxlen = 128
    print("data type: ", type(data), data.shape)
    data = data[:150000]
    print("data type: ", type(data), data.shape)
    torch.cuda.empty_cache()
    data = pad_sequences(data, maxlen=maxlen, padding='post')
    data = torch.from_numpy(data)
    dataloader = torch.utils.data.DataLoader(data, batch_size=opt.batch_size, shuffle=False, num_workers=10)
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
        for step, bz_data in enumerate(dataloader):
            bz_data = bz_data.long().transpose(1, 0).contiguous()
            bz_data = bz_data.to(device)
            optimizer.zero_grad()

            input, target = bz_data[:-1, :], bz_data[1:, :]
            output, _ = net(input)
            loss = criterion(output, target.view(-1))
            loss.backward()
            optimizer.step()

            if step % 5000 == 0:
                print(f'epoch {epoch} | step {step}')
                prefix_words = "what is the problems with you"
                gen_sent = eval(net, ix2word, word2ix, prefix_words)
                print(gen_sent)
                # prefix_words = acrostic_poem
                # torch.save(net.state_dict(), '%s_%s.pth' % (opt.model_prefix, epoch))
                torch.save(net, '%s_%s.pth' % (opt.model_prefix, epoch))



class Config(object):
    lr = 1e-3
    weight_decay = 1e-4
    batch_size = 2
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
    poem = beam_generate(net, ix2word, word2ix, prefix_words=prefix_words, beam_width=10)
    gen_sent = " ".join(poem).replace(' eop', '.')
    return gen_sent



def infer(prefix_words, beam_size, qsize):
    prefix_words = prefix_words.lower()
    corpus_path = './data/tox_com.npz'
    if not os.path.exists(corpus_path):
        read_data_csv(corpus_path)

    d = np.load('./data/tox_com.npz', allow_pickle=True)
    _, word2ix, ix2word = d['data'], d['word2ix'].item(), d['ix2word'].item()

    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    net = Net(len(word2ix), 128, 256)  # TODO: Why 128 and 256?
    net = nn.DataParallel(net)
    checkpoint = './checkpoints/english_4.pth'
    net.load_state_dict(torch.load(checkpoint).state_dict())
    # net = torch.load(checkpoint)
    # prefix_words = None
    # prefix_words = "i feel very tied"
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


# if __name__ == "__main__":
#     # opt = Config()
#     # train(opt)
#     prefix_words = "Life is complicated enough these days"
#     infer(prefix_words)
