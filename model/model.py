import torch
from torch import autograd, nn
import torch.nn.functional as F

num_layers = 7
num_direction = 1
batch_size = 11
hidden_size = 3
emb_dim = 3

hx = autograd.Variable(torch.FloatTensor(num_layers*num_direction,batch_size,
                                                                hidden_size))
cx = autograd.Variable(torch.FloatTensor(num_layers*num_direction,batch_size,
                                                                hidden_size))

class Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.emb = nn.Embedding(input_size, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hidden_size, batch_first=True)
        self.Lin = nn.Linear(hidden_size, num_classes)

    def get_ch(self,size):
        hx = autograd.Variable(torch.FloatTensor(num_layers*num_direction, size,
                                                        hidden_size).zero_())
        cx = autograd.Variable(torch.FloatTensor(num_layers*num_direction, size,
                                                        hidden_size).zero_())
        return (hx,cx)

    def forward(self, inp):
        hc = self.get_ch(inp.size(0))
        e = self.emb(inp)
        l = self.lstm(e, hc)
        y = l[-1] #batch x 1 x hidden
        y = torch.squeeze(y[0])
        return F.softmax(self.Lin(y))
