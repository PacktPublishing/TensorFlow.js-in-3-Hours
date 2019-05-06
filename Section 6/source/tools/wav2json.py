#!/usr/bin/env python3
import librosa
import sys
import numpy as np
from os import path, listdir
import json

def get_idx_from_file(out):
    dir=None
    if out.startswith('03-01-01'):
        dir="01-neutral"
    if out.startswith('03-01-02'):
        dir="02-calm"
    if out.startswith('03-01-03'):
        dir="03-happy"
    if out.startswith('03-01-04'):
        dir="04-sad"
    if out.startswith('03-01-05'):
        dir="05-angry"
    if out.startswith('03-01-06'):
        dir="06-fearful"
    if out.startswith('03-01-07'):
        dir="07-disgust"
    if out.startswith('03-01-08'):
        dir="08-surprised"
    if dir is None:
        raise Exception('Unknown file name scheme %r' % out)
    return int(dir.split('-')[0])

def wav2json(wav, nf, dtype='train'):
    x, _=librosa.load(wav, res_type='kaiser_fast', duration=3, offset=0.5)
    mx=librosa.feature.mfcc(y=x, n_mfcc=25)
    mfccs=np.mean(mx, axis=0)
    mfccs=[-(mfccs/100)]
    mfccs=mfccs[0].tolist()
    fname=path.basename(wav)
    ci=get_idx_from_file(fname)

    i=nf
    outf=path.join('data/json_features', '%d.json' % i)
    print(outf)
    with open(outf, 'w') as out:
        out.write(json.dumps([ci]+mfccs))
    return ci

if __name__ == '__main__':
    dir=sys.argv[1]
    i=0
    for d in listdir(dir):
        if not path.isdir(path.join(dir, d)):
            continue
        for f in listdir(path.join(dir, d)):
            if not f.endswith('.wav'):
                continue
            fin=path.join(dir, d, f)
            i+=1
            wav2json(fin, i)
