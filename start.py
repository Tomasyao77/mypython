import sys
import os
from config import cfg

if __name__ == '__main__':
    f1 = os.listdir(cfg.dataset.morph2)
    f2 = os.listdir(cfg.dataset.morph2)
    print(f1[1000] == f2[1000])