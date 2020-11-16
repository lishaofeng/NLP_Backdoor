import torch
import torch.nn as nn
import torch.nn.functional as F


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