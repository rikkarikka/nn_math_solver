import sys
import os
import random
import itertools
import train
import torch

#print('Current Device:', torch.cuda.current_device())
#device = int(input("Which GPU? "))
torch.cuda.set_device(1)
#print('Current Device:', torch.cuda.current_device())

rand = True

net_type = ('lstm', 'gru')
#lr = (.001, .002)
epochs = 100,
bs = 64,
opt = ('adamax', 'adam', 'sgd')
num_lay =  (1, 2,) #4
hs = (300,)#, 100, 500, 750, 1000, 2000)
num_dir = 2,
embdim = (750, 1000)#, 1250, 1500) #(50, 100, 200, 300, 500,
embfix = (False,)#True)
ptemb = (False,)# True)
dropout = (0, .3, .5, .7)



x = list(itertools.product(net_type, epochs, bs, opt, num_lay, hs, num_dir, embdim, embfix, ptemb, dropout))
if rand: random.shuffle(x)
try:
    for (net_type, epoch, bs, opt, num_lay, hs, num_dir, embdim, embfix, ptemb, dropout) in x:
        if not (embfix and not ptemb):
            print(('Training: (net_type=%s, epoch=%d, bs=%d, opt=%s, num_lay=%d, hs=%d, num_dir=%d, embdim=%d, embfix=%s, ptemb=%s, dropout=%.1f})') %
                (net_type, epoch, bs, opt, num_lay, hs, num_dir, embdim, embfix, ptemb, dropout))

            os.system('python train.py' + \
                    ' -net-type=' + str(net_type) + \
                    #' -lr=' + str(lr) + \
                    ' -epochs=' + str(epochs[0]) + \
                    ' -batch-size=' + str(bs) + \
                    ' -opt=' + opt + \
                    ' -num-layers=' + str(num_lay) + \
                    ' -hidden-sz=' + str(hs) + \
                    ' -num-dir=' + str(num_dir) + \
                    ' -emb-dim=' + str(embdim) + \
                    ' -embfix=' + str(embfix) + \
                    ' -pretr-emb=' + str(ptemb) + \
                    ' -dropout=' + str(dropout) + \
                    ' -device=' + str(device))
            os.system('sort -o ./saved_models/best_models.txt ./saved_models/best_models.txt')
except(KeyboardInterrupt, SystemExit):
    sys.exit("Interrupted by ctrl+c\n")
