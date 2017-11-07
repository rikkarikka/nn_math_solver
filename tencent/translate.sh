#!/bin/bash

python ../OpenNMT-py/translate.py -gpu 0 -model ./models/Math23K-model_acc_89.79_ppl_1.42_e23.pt -src ./data/src-val.txt -tgt ./data/tgt-val.txt -replace_unk -verbose -output ./pred.txt
