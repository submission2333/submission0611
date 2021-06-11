#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
import networkx as nx

from utils import Split_Train_Test

edges = np.loadtxt('data/{}_dataset/{}_data.txt'.format('checkin', 'checkin'))
train_ratio = 0.85
n_nodes = 70
train_edges, test_edges = Split_Train_Test(edges, train_ratio)

snap_1 = test_edges[np.where(test_edges[:, 0]==28)][:, 1:3]
snap_2 = test_edges[np.where(test_edges[:, 0]==29)][:, 1:3]
snap_3 = test_edges[np.where(test_edges[:, 0]==30)][:, 1:3]
snap_4 = test_edges[np.where(test_edges[:, 0]==31)][:, 1:3]

snap_shot = np.zeros((70, 70))

mask = np.zeros_like(snap_shot)
mask[np.triu_indices_from(mask)] = True

for i in snap_1:
    if i[0] != i[1]:
        snap_shot[int(i[0]), int(i[1])] += 1
        snap_shot[int(i[1]), int(i[0])] += 1
    else:
        snap_shot[int(i[0]), int(i[1])] += 1

with sb.axes_style("white"):
    f, ax = plt.subplots(figsize=(7, 5))
    ax = sb.heatmap(snap_shot, mask=mask, vmin=0, vmax=15, cmap=sb.cm.rocket_r)
plt.tight_layout()
plt.show()


for i in snap_2:
    if i[0] != i[1]:
        snap_shot[int(i[0]), int(i[1])] += 1
        snap_shot[int(i[1]), int(i[0])] += 1
    else:
        snap_shot[int(i[0]), int(i[1])] += 1
        

with sb.axes_style("white"):
    f, ax = plt.subplots(figsize=(7, 5))
    ax = sb.heatmap(snap_shot, mask=mask, vmin=0, vmax=15, cmap=sb.cm.rocket_r)
plt.tight_layout()


for i in snap_3:
    if i[0] != i[1]:
        snap_shot[int(i[0]), int(i[1])] += 1
        snap_shot[int(i[1]), int(i[0])] += 1
    else:
        snap_shot[int(i[0]), int(i[1])] += 1
        

with sb.axes_style("white"):
    f, ax = plt.subplots(figsize=(7, 5))
    ax = sb.heatmap(snap_shot, mask=mask, vmin=0, vmax=15, cmap=sb.cm.rocket_r)
plt.tight_layout()


for i in snap_4:
    if i[0] != i[1]:
        snap_shot[int(i[0]), int(i[1])] += 1
        snap_shot[int(i[1]), int(i[0])] += 1
    else:
        snap_shot[int(i[0]), int(i[1])] += 1
        

with sb.axes_style("white"):
    f, ax = plt.subplots(figsize=(7, 5))
    ax = sb.heatmap(snap_shot, mask=mask, vmin=0, vmax=15, cmap=sb.cm.rocket_r)
plt.tight_layout()

