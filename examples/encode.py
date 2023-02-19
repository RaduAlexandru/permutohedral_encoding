#!/usr/bin/env python3

import torch
import permutohedral_encoding as permuto_enc
import numpy as np

#create encoding
pos_dim=3
capacity=262144 #2pow18
nr_levels=24 
nr_feat_per_level=2 
coarsest_scale=1.0 ##we tested that at sigma of 4 is when we slice form just one lattice 
finest_scale=0.0001 #default
scale_list=np.geomspace(coarsest_scale, finest_scale, num=nr_levels)
encoding=permuto_enc.PermutoEncoding(pos_dim, capacity, nr_levels, nr_feat_per_level, scale_list, appply_random_shift_per_level=True, concat_points=True, concat_points_scaling=1.0)

#create points
nr_points=1000
points=torch.rand(nr_points, pos_dim).cuda()

#encode
features=encoding(points)

print("features is ", features.shape)


#loss
loss=features.sum()

loss.backward()

