from tqdm import tqdm
import time
from collections import OrderedDict


def dataloater():
    return [("a", 1), ("b", 2), ("c", 3), ("d", 4), ("e", 5)]


def main():
    text = ""
    # for char in tqdm(["a", "b", "c", "d"]):
    #     time.sleep(0.25)
    #     text = text + char

    with tqdm(dataloater()) as _tqdm:
        for i, (x, y) in enumerate(_tqdm, 1):
            time.sleep(0.5)
            print(x, y)
            _tqdm.set_postfix(OrderedDict(stage="train", epoch=i, loss=0.2), sample_num=64)

    print(text)


if __name__ == "__main__":
    main()
