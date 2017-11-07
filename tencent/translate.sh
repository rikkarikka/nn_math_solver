#!/bin/bash

python ../OpenNMT-py/translate.py -gpu 0 -model ./models/Math23K-model_acc_81.39_ppl_1.65_e11.pt -src ./data/src-val.txt -tgt ./data/tgt-val.txt -replace_unk -verbose -output ./pred.txt
