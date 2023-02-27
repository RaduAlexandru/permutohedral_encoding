#!/usr/bin/env python3

import torch
import permutohedral_encoding as permuto_enc
import numpy as np
import math
import time as time_module

torch.manual_seed(0)


def inference():
    iters_valid=0
    mean_time=0
    min_time=99999999
    with torch.set_grad_enabled(False):
        for iter_nr in range(iters_to_measure):
            #encode
            torch.cuda.synchronize()
            start = time_module.perf_counter()
            features=encoding(points)
            torch.cuda.synchronize()
            elapsed_s=time_module.perf_counter() - start
            # print("elapsed_s",elapsed_s)

            if iter_nr>iters_warmup:
                elapsed_ms=elapsed_s*1000
                mean_time+=elapsed_ms
                min_time=min(min_time, elapsed_ms)
                iters_valid+=1
    print("inference-----")
    print("avg ms", mean_time/iters_valid)
    print("min time", min_time)

def backward_towards_lattice():
    iters_valid=0
    mean_time=0
    min_time=99999999
    with torch.set_grad_enabled(True):
        for iter_nr in range(iters_to_measure):
            #encode
            torch.cuda.synchronize()
            start = time_module.perf_counter()
            features=encoding(points)
            loss=(features.mean()-10).abs()
            loss.backward()
            torch.cuda.synchronize()
            elapsed_s=time_module.perf_counter() - start
            # print("elapsed_s",elapsed_s)

            if iter_nr>iters_warmup:
                elapsed_ms=elapsed_s*1000
                mean_time+=elapsed_ms
                min_time=min(min_time, elapsed_ms)
                iters_valid+=1
    print("train-----")
    print("avg ms", mean_time/iters_valid)
    print("min time", min_time)

def backward_towards_lattice_and_pos():
    iters_valid=0
    mean_time=0
    min_time=99999999
    points_grad=points.clone()
    points_grad.requires_grad_(True)
    with torch.set_grad_enabled(True):
        # points=points.clone()
        for iter_nr in range(iters_to_measure):
            #encode
            torch.cuda.synchronize()
            start = time_module.perf_counter()
            features=encoding(points_grad)
            loss=(features.mean()-10).abs()
            loss.backward()
            torch.cuda.synchronize()
            elapsed_s=time_module.perf_counter() - start
            # print("elapsed_s",elapsed_s)

            if iter_nr>iters_warmup:
                elapsed_ms=elapsed_s*1000
                mean_time+=elapsed_ms
                min_time=min(min_time, elapsed_ms)
                iters_valid+=1
    print("train_and_backpos-----")
    print("avg ms", mean_time/iters_valid)
    print("min time", min_time)

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
iters_to_measure=30
# iters_warmup=1
# iters_to_measure=3

inference()
backward_towards_lattice()
# backward_towards_lattice_and_pos()

