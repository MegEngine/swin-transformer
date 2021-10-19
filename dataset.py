# -*- coding: utf-8 -*-
# Copyright (c) 2021 Megvii Inc.
# Licensed under The MIT License [see LICENSE for details]
import numpy as np
import megengine
import megengine.data as data
import megengine.data.transform as T
from megengine.data.dataset import Dataset
from megengine.data.sampler import RandomSampler, SequentialSampler

class MyDataset(Dataset):
    def __init__(self,args):
        super().__init__()
        self.image = np.random.randn(3, 224, 224).astype("float32")
        self.label = np.random.randint(0, 1000, size=(1,)).astype("int32")

    def __getitem__(self,index):
        return self.image,self.label[0]

    def __len__(self):
        return 60000000

def get_dataloader(args):
    dataset = MyDataset(args)
    sampler = RandomSampler(dataset, batch_size=args.batch_size)
    loader = data.DataLoader(dataset,sampler,num_workers=20,preload=args.preload)
    return loader

