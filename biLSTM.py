import torch.nn as nn
from utils.crf import CRF

class biLSTM_CRF(nn.Module):
    def __init__(self, embedding_size, hidden_size, dict_number, num_labels):
        super(biLSTM_CRF, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size

        self.embedding = nn.Embedding(dict_number, self.embedding_size)
        self.biLSTM = nn.LSTM(self.embedding_size, self.hidden_size, batch_first=True, bidirectional=True, dropout=0.5)
        self.classifer = nn.Linear(self.hidden_size*2, num_labels)
        self.crf = CRF(num_tags=num_labels, batch_first=True)

    def forward(self, x, y, masks):
        x_embed = self.embedding(x)
        output, (final_hidden_state, final_cell_state) = self.biLSTM(x_embed)
        output = self.classifer(output)

        y_pred = self.crf._viterbi_decode(output,masks)

        loss = self.crf.forward(output,y, masks, 'mean')

        return y_pred,loss
