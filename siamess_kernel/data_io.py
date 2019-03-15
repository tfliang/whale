import os
import pandas as pd
import pickle
from tqdm import tqdm
import PIL as pl
import imagehash as imghash
import numpy as np
import math


BASE_PATH = '/home/nod/Projects/data'
TRAIN_DF = os.path.join(BASE_PATH, 'train.csv')
SUB_DF = os.path.join(BASE_PATH, 'sample_submission.csv')
TRAIN =os.path.join(BASE_PATH, 'train')
TEST = os.path.join(BASE_PATH, 'test')

BASE_meta = '/home/nod/Projects/whale/data/metadata'
P2H = os.path.join(BASE_meta, 'p2h.pickle')
P2SIZE = os.path.join(BASE_meta, 'p2size.pickle')
BB_DF = os.path.join(BASE_meta, 'bounding_boxes.csv')

BASE_model = '/home/nod/Projects/whale/piotte'
Piotte_standard = os.path.join(BASE_model, 'mpiotte-standard.model')
Piotte_bootstrap = os.path.join(BASE_model, 'mpiotte-bootstrap.model')


def expand_path(p):
    if os.path.isfile(os.path.join(TRAIN, p)):
        return os.path.join(TRAIN, p)
    if os.path.isfile(os.path.join(TEST, p)):
        return os.path.join(TEST, p)
    return None


class DataIO:
    
    def __init__(self):
        self.tagged, self.submit, self.join = self.load_pids()
        self.p2size = self.load_p2size() 
        self.p2h = self.load_p2hash() 
        self.h2ps = self.load_h2ps()
        self.h2p = self.load_h2p()
        self.p2bb = pd.read_csv(BB_DF).set_index("Image")

    def load_pids(self):
        data = pd.read_csv(TRAIN_DF).to_records()
        tagged = dict([(p, w) for _, p, w in data])
        submit = [p for _, p, _ in pd.read_csv(SUB_DF).to_records()]
        join = list(tagged.keys()) + submit
        return tagged, submit, join

    def load_p2size(self):
        if os.path.isfile(P2SIZE):
            print('p2size exists')
            with open(P2SIZE, 'rb') as f:
                p2size = pickle.load(f)
        else:
            p2size = {}
            for p in tqdm(self.join):
                size = pl.Image.open(expand_path(p)).size
                p2size[p] = size
        return p2size

    def load_p2hash(self):
        if os.path.isfile(P2H):
            print('p2h exists')
            with open(P2H, 'rb') as f:
                p2h = pickle.load(f)
        else:
            p2h = self.compute_p2h()
        return p2h

    def compute_p2h(self):
        p2h = {}
        for p in tqdm(self.join):
            p_path = expand_path(p)
            if p_path:
                try:
                    img = pl.Image.open(p_path)
                    h = imghash.phash(img)
                    p2h[p] = h
                except Exception as e:
                    print(p_path)
                    os.remove(p_path)
                    continue
        h2ps = {}
        for p, h in p2h.items():
            h2ps.setdefault(h, []).append(p)
        hs = list(h2ps.keys())
        h2h = {}
        for i, h1 in enumerate(tqdm(hs)):
            for h2 in hs[:i]:
                ps1 = h2ps[h1]
                ps2 = h2ps[h2]
                if h1 - h2 <= 6 and self.check_match(ps1, ps2):
                    s1 = str(h1)
                    s2 = str(h2)
                    if s1 < s2:
                        s1, s2 = s2, s1
                    h2h[s1] = s2
        for p, h in p2h.items():
            h = str(h)
            p2h[p] = h2h.get(h, h)
        return p2h

    @staticmethod
    def check_match(ps1, ps2):
        for p1 in ps1:
            for p2 in ps2:
                i1 = pl.Image.open(expand_path(p1))
                i2 = pl.Image.open(expand_path(p2))
                if i1.mode != i2.mode or i1.size != i2.size:
                    return False
                a1 = np.array(i1)
                a1 = a1 - a1.mean()
                a1 = a1 / math.sqrt((a1 ** 2).mean())
                a2 = np.array(i2)
                a2 = a2 - a2.mean()
                a2 = a2 / math.sqrt((a2 ** 2).mean())
                a = ((a1 - a2) ** 2).mean()
                if a > 0.1: 
                    return False
        return True

    def load_h2ps(self):
        h2ps = {}
        for p, h in self.p2h.items():
            h2ps.setdefault(p, []).append(h)
        return h2ps

    @staticmethod
    def show_pic(p):
        pf = expand_path(p)
        if pf:
            img = pl.Image.open(pf)
        pass

    @staticmethod
    def read_raw_image(p):
        pf = expand_path(p)
        if pf:
            img = pl.Image.open(pf)
            return img
        return None

    def prefer(self, pics):
        if len(pics) == 1:
            return pics[0]
        best_p = pics[0]
        best_s = self.p2size[best_p]
        for i in range(1, len(pics)):
            p = pics[i]
            s = self.p2size[p]
            if s[0] * s[1] > best_s[0] * best_s[1]:
                best_p = p
                best_s = s
        return best_p

    def load_h2p(self):
        h2p = {}
        for h, pics in self.h2ps.items():
            h2p[h] = self.prefer(pics)
        return h2p


if __name__ == '__main__':
    data = DataIO()
    print(data.p2bb.head())
        







