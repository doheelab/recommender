# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 23:28:45 2020

@author: dohee
"""


# Getting the data

import numpy as np
from lightfm.datasets import fetch_movielens
movielens = fetch_movielens()

for key, value in movielens.items():
    print(key, type(value), value.shape)
    
    
train = movielens['train']
test = movielens['test']


# explicit feedback
train.toarray().shape

# implicit
test.toarray().shape

#%%

# Fitting models

from lightfm import LightFM
from lightfm.evaluation import precision_at_k
from lightfm.evaluation import auc_score

model = LightFM(learning_rate=0.05, loss='bpr')
model = LightFM(learning_rate=0.05, loss='warp')
model.fit(train, epochs=10)

train_precision = precision_at_k(model, train, k=10).mean()
test_precision = precision_at_k(model, test, k=10).mean()

train_auc = auc_score(model, train).mean()
test_auc = auc_score(model, test).mean()

print('Precision: train %.2f, test %.2f.' % (train_precision, test_precision))
print('AUC: train %.2f, test %.2f.' % (train_auc, test_auc))
