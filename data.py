import os
from itertools import islice
from PIL import Image
import cv2
from utils import logger


class Config:
    data_path = '/Users/sid/data/whale'
    train_path = os.path.join(data_path, 'train')
    test_path = os.path.join(data_path, 'test')
    label_file = os.path.join(data_path, 'train.csv')


class Data:

    @classmethod
    def show(cls, name, source='train'):
        if source == 'train':
            fpath = Config.train_path
        else:
            fpath = Config.test_path
        fname = os.path.join(fpath, name)
        logger.info('fname: {}'.format(fname))
        img = Image.open(fname)
        img.show()

    @classmethod
    def show2(cls, name, source='train'):
        if source == 'train':
            fpath = Config.train_path
        else:
            fpath = Config.test_path
        fname = os.path.join(fpath, name)
        logger.info('fname: {}'.format(fname))
        img = cv2.imread(fname, flags=cv2.IMREAD_COLOR)
        print(type(img))
        cv2.imshow('test', img)
        cv2.waitKey()
        cv2.destroyAllWindows()


def test():
    photos = []
    names = []
    rs = dict()
    rs_ct = dict()
    with open(Config.label_file) as f:
        for myline in islice(f, 1, None):
            p, n = myline[:-1].split(',')
            photos.append(p)
            names.append(n)
            rs.setdefault(n, []).append(p)
            rs_ct.setdefault(n, 0)
            rs_ct[n] += 1
    print('p: {}, n: {}'.format(len(photos), len(names)))
    print('p: {}, n: {}'.format(len(set(photos)), len(set(names))))
    rs = sorted(rs.items(), key=lambda x: len(x[1]))
    print(rs[-2])


if __name__ == '__main__':
    test()
    #Data.show(name='02bdec750.jpg')
