import os
import pandas as pd


BASE_PATH = '/home/tfliang/whale_data/data'
TRAIN_DF = os.path.join(BASE_PATH, 'train.csv')
SUB_Df = os.path.join(BASE_PATH, 'sample_submission.csv')
TRAIN =os.path.join(BASE_PATH, 'train')
TEST = os.path.join(BASE_PATH, 'test')

BASE_meta = '/home/tfliang/whale/data/metadata'
P2H = os.path.join(BASE_meta, 'p2h.pickle')
P2SIZE = os.path.join(BASE_meta, 'p2size.pickle')
BB_DF = os.path.join(BASE_meta, 'bounding_boxes.csv')

BASE_model = '/home/tfliang/whale/piotte'
Piotte_standard = os.path.join(BASE_model, 'mpiotte-standard.model')
Piotte_bootstrap = os.path.join(BASE_model, 'mpiotte-bootstrap.model')


class DataIO:
    
    def __init__(self):
        pass

    @staticmethod
    def load_pids():
        data = pd.read_csv(TRAIN_DF).to_records()
        print(data)


if __name__ == '__main__':
    DataIO.load_pids()
        







