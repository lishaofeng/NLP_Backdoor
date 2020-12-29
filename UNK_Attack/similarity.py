import torch
import torch.nn as nn

def euclide_dist(x, y):
    dist = torch.pow(x - y, 2).sum()
    return dist

def cos_sim(input1, input2):
    cos = nn.CosineSimilarity(dim=2, eps=1e-6)
    output = cos(input1, input2)

    return output
