#!/usr/bin/env python3

import torch
import permutohedral_encoding as permuto_enc
import numpy as np

#create encoding
pos_dim=3
capacity=pow(2,18) 
nr_levels=24 
nr_feat_per_level=2 
coarsest_scale=1.0 
finest_scale=0.0001 
scale_list=np.geomspace(coarsest_scale, finest_scale, num=nr_levels)
encoding=permuto_enc.PermutoEncoding(pos_dim, capacity, nr_levels, nr_feat_per_level, scale_list)

#create mlp which processes the encoded points and gives the output
mlp= torch.nn.Sequential(
        torch.nn.Linear(encoding.output_dims() ,32),
        torch.nn.GELU(),
        torch.nn.Linear(32,32),
        torch.nn.GELU(),
        torch.nn.Linear(32,32),
        torch.nn.GELU(),
        torch.nn.Linear(32,1)
    ).cuda()

#a coarse to fine optimization which anneals the coarse levels of the grid before it starts adding the higher resolution ones
nr_iters_for_c2f=1000
c2f=permuto_enc.Coarse2Fine(nr_levels)

#optimizer
params = list(encoding.parameters()) + list(mlp.parameters()) 
optimizer=torch.optim.AdamW(params, lr=1e-4)

#create points
nr_points=1000
points=torch.rand(nr_points, pos_dim).cuda()

iter_nr=0
while True:

    #create a 1D annealing window that performs coarse-to-fine optimization
    window=c2f( permuto_enc.map_range_val(iter_nr, 0.0, nr_iters_for_c2f, 0.3, 1.0   ) )

    #encode
    features=encoding(points,window)

    #mlp
    output=mlp(features)

    #loss
    target=10.0
    loss=(output-target).mean().abs()
    if(iter_nr%100==0):
        print("loss is ", loss)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


    iter_nr+=1


