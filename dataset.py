import pickle
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

def build_dataset(x_path, y_path):

    with open(x_path, 'rb') as f:
        x = pickle.load(f)
    with open(y_path, 'rb') as f:
        y = pickle.load(f)



    masks = [[1 for i in range(len(sen))] for sen in x]

    x_pad = pad_sequence([torch.LongTensor(np.array(i)) for i in x], batch_first=True)
    y = pad_sequence([torch.LongTensor(np.array(i)) for i in y], batch_first=True)
    masks = pad_sequence([torch.ByteTensor(np.array(i)) for i in masks], batch_first=True)


    return TensorDataset(x_pad, y, masks)

