#!/usr/bin/env python3

import torch
import numpy as np
import math
import time as time_module
import permutohedral_encoding as permuto_enc
import tinycudann as tcnn

torch.manual_seed(0)
torch.set_default_tensor_type(torch.cuda.FloatTensor)


class ModelForPerformance(torch.nn.Module):

    def __init__(self, input_channels, log2_hashmap_size, nr_lattice_features, nr_resolutions):
        super(ModelForPerformance, self).__init__()



        

        #encoding voxel
        config_encoding={
            "otype": "HashGrid",
            "n_levels": nr_resolutions,
            "n_features_per_level": nr_lattice_features,
            "log2_hashmap_size": log2_hashmap_size,
            "base_resolution": 4,
            "per_level_scale": 1.5,
            "interpolation": "Linear"
        }
        self.encoding_voxel = tcnn.Encoding(n_input_dims=input_channels, encoding_config=config_encoding)


        #encoding permuto
        pos_dim=input_channels
        capacity=pow(2,log2_hashmap_size) 
        nr_levels=nr_resolutions
        nr_feat_per_level=nr_lattice_features 
        coarsest_scale=1.0 
        finest_scale=0.0001 
        scale_list=np.geomspace(coarsest_scale, finest_scale, num=nr_levels)
        self.encoding_permuto=permuto_enc.PermutoEncoding(pos_dim, capacity, nr_levels, nr_feat_per_level, scale_list) 
        

    
    def forward(self, points, lattice_type ):

        if(lattice_type=="permuto"):
            points_encoded=self.encoding_permuto(points)
        else:
            points_encoded=self.encoding_voxel(points) 
       
       
        return points_encoded

        
def inference(model, pts, lattice_type, iters_warmup, iters_to_measure):
    nr_points=pts.shape[0]
    pos_dim=pts.shape[1]
    model.eval()
    with torch.set_grad_enabled(False):
        mean_inference_throughput=0
        nr_iterations_executed=0
        with torch.set_grad_enabled(False):
            for iter_nr in range(iters_to_measure):
                torch.cuda.synchronize()
                start = time_module.perf_counter()

                pred=model(pts, lattice_type)

                torch.cuda.synchronize()
                elapsed_s=time_module.perf_counter() - start


                if iter_nr>iters_warmup:
                    throughput = nr_points / elapsed_s
                    mean_inference_throughput+=throughput
                    nr_iterations_executed+=1
            mean_inference_throughput=int(mean_inference_throughput/nr_iterations_executed)
            print("mean_inference_throughput ", lattice_type, " pos_dim ", pos_dim, " " , f"{mean_inference_throughput:,}")

def training_backward_towards_lattice(model, pts, lattice_type, iters_warmup, iters_to_measure, optimizer):
    nr_points=pts.shape[0]
    pos_dim=pts.shape[1]
    model.train()
    with torch.set_grad_enabled(True):
        mean_training_throughput=0
        nr_iterations_executed=0
        for iter_nr in range(iters_to_measure):
            torch.cuda.synchronize()
            start = time_module.perf_counter()

            pred=model(pts, lattice_type)
            loss=((pred-10)**2).mean()

            #backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            torch.cuda.synchronize()
            elapsed_s=time_module.perf_counter() - start

        
            if iter_nr>iters_warmup:
                throughput = nr_points / elapsed_s
                mean_training_throughput+=throughput
                nr_iterations_executed+=1
        mean_training_throughput=int(mean_training_throughput/nr_iterations_executed)
        print("mean_training_backwards_lattice_throughput ", lattice_type, " pos_dim ", pos_dim, " " , f"{mean_training_throughput:,}")
        # print("loss is", loss.item())

def training_backward_towards_lattice_and_pos(model, pts, lattice_type, iters_warmup, iters_to_measure, optimizer):
    nr_points=pts.shape[0]
    pos_dim=pts.shape[1]
    model.train()
    points_grad=pts.clone()
    points_grad.requires_grad_(True)
    with torch.set_grad_enabled(True):
        mean_training_throughput=0
        nr_iterations_executed=0
        for iter_nr in range(iters_to_measure):
            torch.cuda.synchronize()
            start = time_module.perf_counter()

            pred=model(points_grad, lattice_type)
            loss=((pred-10)**2).mean()

            #backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            torch.cuda.synchronize()
            elapsed_s=time_module.perf_counter() - start
        

            if iter_nr>iters_warmup:
                throughput = nr_points / elapsed_s
                mean_training_throughput+=throughput
                nr_iterations_executed+=1
        mean_training_throughput=int(mean_training_throughput/nr_iterations_executed)
        print("mean_training_backwards_lattice_and_pos_throughput ", lattice_type, " pos_dim ", pos_dim, " " , f"{mean_training_throughput:,}")
        # print("loss is", loss.item())


def measure_performance(nr_points, pos_dim, lattice_type):

    #params
    nr_lattice_features=2
    nr_resolutions=24
    log2_hashmap_size=18
    


    #model
    model=ModelForPerformance(pos_dim, log2_hashmap_size, nr_lattice_features, nr_resolutions)

    optimizer = torch.optim.AdamW (model.parameters(), amsgrad=False,  betas=(0.9, 0.99), eps=1e-15, weight_decay=0.0, lr=1e-3)

    #create some random points for input
    pts=torch.rand(nr_points, pos_dim) #range 0,1, we want it in [0.5,0.5]
    pts=pts-0.5

    #params
    iters_warmup=10
    iters_to_measure=300
    
    inference(model, pts, lattice_type, iters_warmup, iters_to_measure) 
    training_backward_towards_lattice(model, pts, lattice_type, iters_warmup, iters_to_measure, optimizer)
    training_backward_towards_lattice_and_pos(model, pts, lattice_type, iters_warmup, iters_to_measure, optimizer)



def run():

    dims=[2,3,4,5,6]
    for pos_dim in dims:
        print("")
        measure_performance( int(math.pow(2,19)), pos_dim, "permuto") 
        measure_performance( int(math.pow(2,19)), pos_dim, "voxel") 


    return



def main():
    run()

if __name__ == "__main__":
    main()  # This is what you would have, but the following is useful:

    # # These are temporary, for debugging, so meh for programming style.
    # import sys, trace

    # # If there are segfaults, it's a good idea to always use stderr as it
    # # always prints to the screen, so you should get as much output as
    # # possible.
    # sys.stdout = sys.stderr

    # # Now trace execution:
    # tracer = trace.Trace(trace=1, count=0, ignoredirs=["/usr", sys.prefix])
    # tracer.run('main()')


