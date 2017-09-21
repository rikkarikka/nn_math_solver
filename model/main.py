import sys
import os
import itertools
import train

lr = (.001, .002)
epochs = 100,
bs = 64,
opt = ('adamax', 'adam', 'sgd')
num_lay = (1, 2, 4)
hs = (100, 300, 500, 750, 1000, 2000)
num_dir = 2,
embdim = (50, 100, 200, 300)
embfix = (False,)# True)
ptemb = (True, False)

x = list(itertools.product(lr, epochs, bs, opt, num_lay, hs, num_dir, embdim, embfix, ptemb))
try:
    for (lr, epoch, bs, opt, num_lay, hs, num_dir, embdim, embfix, ptemb) in x:
        print(('Training: (lr=%f, epoch=%d, bs=%d, opt=%s, num_lay=%d, hs=%d, num_dir=%d, embdim=%d, embfix=%s, ptemb=%s)') %
        (lr, epoch, bs, opt, num_lay, hs, num_dir, embdim, embfix, ptemb))
        os.system('python train.py' + \
                    ' -lr=' + str(lr) + \
                    ' -epochs=' + str(epochs[0]) + \
                    ' -batch-size=' + str(bs) + \
                    ' -opt=' + opt + \
                    ' -num-layers=' + str(num_lay) + \
                    ' -hidden-sz=' + str(hs) + \
                    ' -num-dir=' + str(num_dir) + \
                    ' -emb-dim=' + str(embdim) + \
                    ' -embfix=' + str(embfix) + \
                    ' -pretr-emb=' + str(ptemb))
except(KeyboardInterrupt, SystemExit):
    sys.exit("Interrupted by ctrl+c\n")
