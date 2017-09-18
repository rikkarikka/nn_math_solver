import torch
from torch import autograd, nn
import torch.nn.functional as F

num_layers = 1
num_direction = 2
batch_size = 64
hidden_size = 300
emb_dim = 300

hx = autograd.Variable(torch.FloatTensor(num_layers*num_direction,batch_size,
                                                                hidden_size))
cx = autograd.Variable(torch.FloatTensor(num_layers*num_direction,batch_size,
                                                                hidden_size))

class Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, prevecs=None, embfix=False):
        super().__init__()
        self.emb = nn.Embedding(input_size, emb_dim)
        if prevecs is not None:
          self.emb.weight = nn.Parameter(prevecs)
        if embfix:
          self.emb.weight.requires_grad=False
        self.lstm = nn.LSTM(emb_dim, hidden_size, batch_first=True,bidirectional=(num_direction==2))
        self.Lin = nn.Linear(hidden_size*num_direction, num_classes)

    def get_ch(self,size):
        hx = autograd.Variable(torch.FloatTensor(num_layers*num_direction,
                                                    size, hidden_size).zero_())
        cx = autograd.Variable(torch.FloatTensor(num_layers*num_direction,
                                                    size, hidden_size).zero_())
        if int(torch.cuda.is_available()) == 1:
            hx.data = hx.data.cuda()
            cx.data = cx.data.cuda()
        return (hx,cx)

    def forward(self, inp):
        hc = self.get_ch(inp.size(0))
        e = self.emb(inp)
        _, (y,_) = self.lstm(e, hc)
        if num_direction==2:
          y = torch.cat([y[0:y.size(0):2], y[1:y.size(0):2]], 2)
        y = torch.squeeze(y,0)
        #y = y[-1]
        return self.Lin(y)
