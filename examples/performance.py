#!/usr/bin/env python3

import torch
import permutohedral_encoding as permuto_enc
import numpy as np
import math
import time as time_module

torch.manual_seed(0)

#create encoding
pos_dim=3
capacity=pow(2,18) 
nr_levels=24 
nr_feat_per_level=2 
coarsest_scale=1.0 
finest_scale=0.0001 
scale_list=np.geomspace(coarsest_scale, finest_scale, num=nr_levels)
encoding=permuto_enc.PermutoEncoding(pos_dim, capacity, nr_levels, nr_feat_per_level, scale_list)


#create points
nr_points=int(math.pow(2,19))
points=torch.rand(nr_points, pos_dim).cuda()

iters_warmup=10
iters_to_measure=300
iters_valid=0
mean_time=0
min_time=99999999
for iter_nr in range(iters_to_measure):


    #encode
    torch.cuda.synchronize()
    start = time_module.perf_counter()
    features=encoding(points)
    torch.cuda.synchronize()
    elapsed_s=time_module.perf_counter() - start

    if iter_nr>iters_warmup:
        # throughput = nr_points / elapsed_s
        # mean_training_throughput+=throughput
        # training_nr_iterations_executed+=1
        elapsed_ms=elapsed_s*1000
        mean_time+=elapsed_ms
        min_time=min(min_time, elapsed_ms)
        iters_valid+=1
print("avg ms", mean_time/iters_valid)
print("min time", min_time)

    


