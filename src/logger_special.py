import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import json
import pickle
from torch import save as tsave
import torch
from .utils import create_dir
from datetime import datetime


class LoggerSpecial:
    def __init__(self, algorithm):

        # Logger saves data
        # Updates eps x steps x feature_num
        # Saves at special w/ name algorithm date 
        ## csv and description

        self.features = []
        self.steps = []

        fn_date = datetime.now().strftime("_%m%d_%H-%M-%S")

        self.save_special_path = os.path.join("../specials", algorithm, fn_date)
        create_dir(self.save_special_path)

    def set_description(self, comment):

        description = {
            'comment': comment
            }

        fn = os.path.join(self.save_special_path, 'description.pth' )
        out_file = open(fn,'w+')

        json.dump(description,out_file)


    def update(self, step, feature):
        self.steps.append(step)
        self.features.append(feature)


    def consolidate(self, episode):
        folder = os.path.join(self.save_special_path, 'e{}_n{}'.format(episode, self.steps[-1]))
        create_dir(folder) 

        fn = os.path.join(folder, 'data.csv')

        self.features = map(list, zip(*self.features))
        d = {
            'steps': self.steps,
            }

        for i, feat in enumerate(self.features):
            d['f{}'.format(i)] = feat

        df = pd.DataFrame(d)

        df.to_csv(fn, mode = 'w', index = False)

        self.steps = []
        self.features = []


