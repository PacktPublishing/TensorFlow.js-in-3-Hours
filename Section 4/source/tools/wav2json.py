#!/usr/bin/env python3
# Based on ideas from 
# https://github.com/MITESHPUTHRANNEU/Speech-Emotion-Analyzer
import librosa
import sys
import numpy as np
from os import path, listdir
import json

def wav2json(wav, json_out, dtype='train'):
    x, _=librosa.load(wav, res_type='kaiser_fast', duration=3, offset=0.5)
    mx=librosa.feature.mfcc(y=x, n_mfcc=25)
    mfccs=np.mean(mx, axis=0)
    mfccs=[-(mfccs/100)]
    mfccs=mfccs[0].tolist()
    json_out=open(json_out,"w")
    return json.dump(mfccs, json_out)

if __name__ == '__main__':
    from sys import argv
    out=wav2json(argv[1], argv[2])
    
