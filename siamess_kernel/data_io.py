import os
import pandas as pd
import pickle
from tqdm import tqdm
import PIL as pl


BASE_PATH = '/home/tfliang/whale_data/data'
TRAIN_DF = os.path.join(BASE_PATH, 'train.csv')
SUB_DF = os.path.join(BASE_PATH, 'sample_submission.csv')
TRAIN =os.path.join(BASE_PATH, 'train')
TEST = os.path.join(BASE_PATH, 'test')

BASE_meta = '/home/tfliang/whale/data/metadata'
P2H = os.path.join(BASE_meta, 'p2h.pickle')
P2SIZE = os.path.join(BASE_meta, 'p2size.pickle')
BB_DF = os.path.join(BASE_meta, 'bounding_boxes.csv')

BASE_model = '/home/tfliang/whale/piotte'
Piotte_standard = os.path.join(BASE_model, 'mpiotte-standard.model')
Piotte_bootstrap = os.path.join(BASE_model, 'mpiotte-bootstrap.model')


def expand_path(p):
    if os.path.isfile(os.path.join(TRAIN, p)):
        return os.path.join(TRAIN, p)
    if os.path.isfile(os.path.join(TEST, p)):
        return os.path.join(TEST, p)
    return p


class DataIO:
    
    def __init__(self):
        self.tagged, self.submit, self.join = self.load_pids()
        self.p2size = self.load_p2size() 

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
                size = pl.image.open(expand_path(p)).size
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
        





if __name__ == '__main__':
    data = DataIO()
    print(data.p2size)
        







