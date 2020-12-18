# -*- coding: utf-8 -*-

from __future__ import print_function

import pandas as pd
import os

DB_TRAIN = '../../data/train'
DB_VALIDATION = '../../data/validation'
DB_TEST = '../../data/test'
LABELS = 'data.csv'


class Database(object):

    def __init__(self, db_path: str = DB_TRAIN, labels: str = LABELS):
        self.labels = labels
        self.db_path = db_path

        self._gen_csv()
        self.data = pd.read_csv(self.labels)
        self.classes = set(self.data["cls"])

    def _gen_csv(self):
        if os.path.exists(self.labels):
            return
        with open(self.labels, 'w', encoding='UTF-8') as f:
            f.write("img,cls")
            for root, _, files in os.walk(self.db_path, topdown=False):
                cls = root.split('/')[-1]
                for name in files:
                    if not name.endswith('.jpg'):
                        continue
                    img = os.path.join(root, name)
                    f.write("\n{},{}".format(img, cls))

    def __len__(self):
        return len(self.data)

    def get_class(self):
        return self.classes

    def get_data(self):
        return self.data


if __name__ == "__main__":
    db = Database()
    data = db.get_data()
    classes = db.get_class()

    print("DB length:", len(db))
    print(classes)
