#pragma once

// #include "instant_ngp_2/kernels/HashTableGPU.cuh"

// #include "instant_ngp_2/jitify_helper/jitify_helper.cuh"
#include <torch/torch.h>
// #include <cuda.h>
// #include <cuda_runtime.h>
// #include <cuda_runtime_api.h>
// #include "device_launch_parameters.h" //needed for threadIdx and blockDim 
// #include <torch/torch.h>
// #include <cuda_bf16.h>
// #include <cuda_fp16.h>

//Add this header after we add all cuda stuff because we need the profiler to have cudaDeviceSyncronize defined
// #define ENABLE_CUDA_PROFILING 1
// #include "Profiler.h" 


#define BLOCK_SIZE 128
#define BLOCK_SIZE_BACK 32

#define LATTICE_HALF_PRECISION 0

// class LatticeGPU { 
// public:

//     //during nvrtc compilation we do not want to compile this code as it does not work due to not including vector definition
//     LatticeGPU(){
//         // create_program_handles();
//         // cudaEventCreate (&m_event_nr_vertices_lattice_changed);
//     }

//     // //it uses Jittify to get a handle for the programs. The programs can contain more than one kernel.. It doesnt actually compile them, they will get jit compiled the first time you run them
//     // void create_program_handles(){
//     //     m_lattice_program=create_jitify_program( std::string(CMAKE_SOURCE_DIR)+"/include/instant_ngp_2/kernels/LatticeGPU.cuh" );
//     // }


//     // void slice_with_collisions_standalone_no_precomputation(const float* positions, float* sliced_values, const int pos_dim, const int val_dim, const int nr_positions, int* splatting_indices, float* splatting_weights,  const HashTableGPU& hash_table_gpu){

//     //     dim3 blocks((nr_positions - 1) / BLOCK_SIZE + 1, 1, 1);
//     //     dim3 blockSize(BLOCK_SIZE, 1, 1);
//     //     blockSize.y = 1;

//     //     // CUresult res = slice_with_collisions_no_precomputation<3><<<blocks, blockSize>>>(3, positions, sliced_values, nr_positions, splatting_indices, splatting_weights, hash_table_gpu);

//     //     // CUresult res= m_lattice_program.kernel("slice_with_collisions_no_precomputation")
//     //     //             .instantiate(pos_dim, val_dim)
//     //     //             .configure(blocks, blockSize)
//     //     //             .launch( positions, sliced_values, nr_positions, splatting_indices, splatting_weights, hash_table_gpu);
//     //     // CUDA_CHECK_CURESULT(res);
//     //     CUDA_CHECK_ERROR();

//     // }

    

//     // void slice_backwards_standalone_with_precomputation_no_homogeneous(float* grad_sliced_values, int* splatting_indices, float* splatting_weights,  const int pos_dim, const int val_dim, const int nr_positions, const HashTableGPU& hash_table_gpu){


//     //     dim3 blocks((nr_positions - 1) / BLOCK_SIZE + 1, 1, 1);
//     //     dim3 blockSize(BLOCK_SIZE, 1, 1);
//     //     CUresult res= m_lattice_program.kernel("slice_backwards_with_precomputation_no_homogeneous")
//     //                 .instantiate(pos_dim, val_dim)
//     //                 .configure(blocks, blockSize)
//     //                 .launch( nr_positions, grad_sliced_values, splatting_indices, splatting_weights, hash_table_gpu );
//     //     CUDA_CHECK_CURESULT(res);
//     //     CUDA_CHECK_ERROR()

//     // }

    

//     // void elevate(float* positions, const int pos_dim, const int nr_positions, float* elevated){

//     //     dim3 blocks((nr_positions - 1) / BLOCK_SIZE + 1, 1, 1);
//     //     dim3 blockSize(BLOCK_SIZE, 1, 1);
//     //     CUresult res= m_lattice_program.kernel("elevate_points")
//     //                 .instantiate(pos_dim)
//     //                 .configure(blocks, blockSize)
//     //                 .launch( nr_positions, positions, elevated );
//     //     CUDA_CHECK_CURESULT(res);
//     //     CUDA_CHECK_ERROR()

//     // }

//     // void wait_to_create_vertices(){
//     //     cudaEventSynchronize(m_event_nr_vertices_lattice_changed);
//     // }
    

//     // jitify::Program m_lattice_program;

//     //for syncronization
//     // cudaEvent_t m_event_nr_vertices_lattice_changed; //when doing splatting, distribute, or a create_coarse_verts, we must record this even after the kernel. Afterwards when we call nr_lattice_vertices we wait for this event to have finished indicating that the kernel has finished



   

// };



// //elevated a vector from m_pos_dim to a m_pos_dim+1 space
// template<int pos_dim>
// __device__ void elevate(float* elevated, const float* position){
//     //TODO the scale factor can be precomputed
//     float scaleFactor[pos_dim];
//     // float invStdDev = (pos_dim + 1) * sqrt(2.0f / 3);
//     float invStdDev = 1.0;
//     for (int i = 0; i < pos_dim; i++) {
//         scaleFactor[i] = 1.0f / (sqrt((float) (i + 1) * (i + 2))) * invStdDev;
//     }

//     // embed position vector into the hyperplane
//     // first rotate position into the (pd+1)-dimensional hyperplane
//     // sm contains the sum of 1..n of our feature vector
//     float sm = 0;
//     for (int i = pos_dim; i > 0; i--) {
//         float cf = position[i - 1] * scaleFactor[i - 1];
//         // float cf = position[i - 1] ;
//         elevated[i] = sm - i * cf;
//         sm += cf;
//     }
//     elevated[0] = sm;

// }

// //check if a vector has all coordinates integers like 1.0 or do they have a fractional part. Useful when having lattice keys from different scales
// __device__ bool are_all_coords_integer(const float* vec, const int vec_size){
//     for (int i = 0; i < vec_size; i++) {
//         float val=vec[i];
//         float integer_part=0.0;
//         float decimal_part=0.0;
//         decimal_part=modff(val, &integer_part);
//         decimal_part=fabs(decimal_part);
//         if( decimal_part>0.0001 ){ 
//             return false;
//         }
//     }
//     return true;

// }


// //I believe that when we embedd a fine lattice in a coarse one we can end up with keys of type 0.5, 0.5, -1.0 so making movements of 0.5 with them will end up in non integer keys. This helps me debug this 
// __device__ bool is_only_one_coord_integer(const float* vec, const int vec_size){
//     int nr_integer_coords=0;
//     for (int i = 0; i < vec_size; i++) {
//         float val=vec[i];
//         float integer_part=0.0;
//         float decimal_part=0.0;
//         decimal_part=modff(val, &integer_part);
//         decimal_part=fabs(decimal_part);
//         if( decimal_part<0.0001 ){  //if the decimal part is very small we say it's integer
//             return nr_integer_coords++;
//         }
//     }

//     if (nr_integer_coords==1){
//         return true;
//     }else{
//         return false;
//     }

// }
// __device__ int nr_coords_integer(const float* vec, const int vec_size){
//     int nr_integer_coords=0;
//     for (int i = 0; i < vec_size; i++) {
//         float val=vec[i];
//         float integer_part=0.0;
//         float decimal_part=0.0;
//         decimal_part=modff(val, &integer_part);
//         decimal_part=fabs(decimal_part);
//         if( decimal_part<0.0001 ){  //if the decimal part is very small we say it's integer
//             return nr_integer_coords++;
//         }
//     }

//     return nr_integer_coords;

// }


// template<int pos_dim>
// __global__ void 
// __launch_bounds__(BLOCK_SIZE) //since the block size is known at compile time we can specify it to the kernel and therefore cuda doesnt need to use heuristics based on code complexity to minimize registry usage
// elevate_points(const int nr_positions,  const float* positions, float* elevated){

//     // determine where in the thread grid we are
//     int idx = blockIdx.x * blockDim.x + threadIdx.x; //each thread will deal with a new value

//     if(idx>=nr_positions){ //don't go out of bounds
//         return;
//     }
    
//     float* elevated_point = elevated + idx * (pos_dim + 1);
//     const float *position = positions + idx * pos_dim;
//     elevate<pos_dim>(elevated_point, position);

// }

//fast atomic add for half tensor like https://github.com/pytorch/pytorch/blob/b47ae9810c1a645f4942737ab4a58b2b1407e7bd/aten/src/ATen/native/cuda/KernelUtils.cuh
template <
    typename scalar_t,
    typename index_t,
    typename std::enable_if<std::is_same<c10::Half, scalar_t>::value>::type* =
        nullptr>
__device__ __forceinline__ void fastSpecializedAtomicAdd(
    scalar_t* tensor,
    index_t index,
    const index_t numel,
    scalar_t value) {
#if (                      \
    (defined(USE_ROCM)) || \
    (defined(CUDA_VERSION) && (CUDA_VERSION < 10000)) || \
    (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 700)))
  atomicAdd(
      reinterpret_cast<at::Half*>(tensor) + index,
      static_cast<at::Half>(value));
#else
  // Accounts for the chance tensor falls on an odd 16 bit alignment (ie, not 32 bit aligned)
  __half* target_addr = reinterpret_cast<__half*>(tensor + index);
  bool low_byte = (reinterpret_cast<std::uintptr_t>(target_addr) % sizeof(__half2) == 0);

  if (low_byte && index < (numel - 1)) {
    __half2 value2;
    value2.x = value;
    value2.y = __int2half_rz(0);
    atomicAdd(reinterpret_cast<__half2*>(target_addr), value2);

  } else if (!low_byte && index > 0) {
    __half2 value2;
    value2.x = __int2half_rz(0);
    value2.y = value;
    atomicAdd(reinterpret_cast<__half2*>(target_addr - 1), value2);

  } else {
    atomicAdd(
        reinterpret_cast<__half*>(tensor) + index, static_cast<__half>(value));
  }
#endif
}

// https://stackoverflow.com/a/466278
__device__ __forceinline__ constexpr unsigned long upper_power_of_two(unsigned long val){
    unsigned long v=val;
    v=v-1;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v=v+1;
    return v;


    // return pow(2, ceil(log(val)/log(2)));
}




// template<int pos_dim, int val_dim>
// __global__ void 
// // __launch_bounds__(BLOCK_SIZE) //since the block size is known at compile time we can specify it to the kernel and therefore cuda doesnt need to use heuristics based on code complexity to minimize registry usage
// slice_with_collisions_no_precomputation(
//     const int nr_positions,
//     const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> positions,
//     torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> sliced_values,
//     torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> splatting_indices,
//     torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> splatting_weights,
//     const bool should_precompute_tensors_for_backward,
//     HashTableGPU hash_table,
//     torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> values
//     ) {

//     int idx = blockIdx.x * blockDim.x + threadIdx.x; //each thread will deal with a new value


//     if(idx>=nr_positions){ //don't go out of bounds
//         return;
//     }


    
//     float elevated[pos_dim + 1];

//     //calculate elevated
//     float scaleFactor[pos_dim];
//     // float invStdDev = (pos_dim + 1) * sqrt(2.0f / 3);
//     float invStdDev = 1.0;
//     for (int i = 0; i < pos_dim; i++) {
//         scaleFactor[i] = 1.0f / (sqrt((float) (i + 1) * (i + 2))) * invStdDev;
//     }
//     // embed position vector into the hyperplane
//     // first rotate position into the (pd+1)-dimensional hyperplane
//     // sm contains the sum of 1..n of our feature vector
//     float sm = 0;
//     for (int i = pos_dim; i > 0; i--) {
//         // float cf = position[i*nr_positions - 1*nr_positions] * scaleFactor[i - 1];
//         float cf = positions[idx][i-1] * scaleFactor[i - 1];
//         // float cf = position[i - 1] ;
//         elevated[i] = sm - i * cf;
//         sm += cf;
//     }
//     elevated[0] = sm;


//     int rem0[pos_dim + 1];
//     int rank[pos_dim + 1]{0};


   


//     // Find the closest 0-colored simplex through rounding
//     // greedily search for the closest zero-colored lattice point
//     int sum = 0;
//     for (int i = 0; i <= pos_dim; i++) {
//         float v = elevated[i] * (1.0 / (pos_dim + 1));
//         float up = ceil(v) * (pos_dim + 1);
//         float down = floor(v) * (pos_dim + 1);
//         if (up - elevated[i] < elevated[i] - down) {
//             rem0[i] = (int) up;
//         } else {
//             rem0[i] = (int) down;
//         }
//         sum += rem0[i];
//     }
//     sum /= pos_dim + 1;


//     // Find the simplex we are in and store it in rank (where rank describes what position coordinate i has in the sorted order of the features values)
//     for (int i = 0; i < pos_dim; i++) {
//         double di = elevated[i] - rem0[i];
//         for (int j = i + 1; j <= pos_dim; j++)
//             if (di < elevated[j] - rem0[j])
//                 rank[i]++;
//             else
//                 rank[j]++;
//     }

//     // If the point doesn't lie on the plane (sum != 0) bring it back
//     for (int i = 0; i <= pos_dim; i++) {
//         rank[i] += sum;
//         if (rank[i] < 0) {
//             rank[i] += pos_dim + 1;
//             rem0[i] += pos_dim + 1;
//         } else if (rank[i] > pos_dim) {
//             rank[i] -= pos_dim + 1;
//             rem0[i] -= pos_dim + 1;
//         }
//     }



//     float barycentric[pos_dim + 2]{0};
//     // Compute the barycentric coordinates (p.10 in [Adams etal 2010])
//     for (int i = 0; i <= pos_dim; i++) {
//         float delta = (elevated[i] - rem0[i]) * (1.0 / (pos_dim + 1));
//         barycentric[pos_dim - rank[i]] += delta;
//         barycentric[pos_dim + 1 - rank[i]] -= delta;
//     }
//     // Wrap around
//     barycentric[0] += 1.0 + barycentric[pos_dim + 1];

//     //smoothstep of the linear interpolation like done in the instant ngp paper
//     // for (int i = 0; i <= pos_dim; i++) {
//         // barycentric[i] = barycentric[i]*barycentric[i] *(3-2*barycentric[i]);
//         // float smooth_weight = barycentric[i]*barycentric[i] *(3-2*barycentric[i]);
//         // printf("prev weight is %f, new weight is %f \n", barycentric[i], smooth_weight);
//         //invere smoothstep https://stackoverflow.com/questions/28740544/inverted-smoothstep
//         // if(barycentric[i]>0 && barycentric[i]<1){
//             // barycentric[i] = 0.5 - sin(asin(1.0-2.0*barycentric[i])/3.0);
//         // }
//     // }

//     //from higher order barycentric coordinates https://domino.mpi-inf.mpg.de/intranet/ag4/ag4publ.nsf/3b7127147beb1437c125675300686244/637fcbb7f3f5a70fc12573cc00458c99/$FILE/paper.pdf
//     //same as instant ngp
//     // for (int i = 0; i <= pos_dim; i++) {
//         // barycentric[i] = -2* barycentric[i]*barycentric[i] *( barycentric[i] -3.0/2 );
//         // barycentric[i] = barycentric[i]*barycentric[i] *(3-2*barycentric[i]);
//     // }
//     // //renormalize
//     // float sum_bar=0.0;
//     // for (int i = 0; i <= pos_dim; i++) {
//         // sum_bar+=barycentric[i];
//     // }
//     // printf("sum %f \n", sum_bar);
//     // for (int i = 0; i <= pos_dim; i++) {
//         // barycentric[i]/=sum_bar;
//     // }



//     //here we accumulate the values and the homogeneous term
//     float val_hom[val_dim]{0};

//     int key[pos_dim];
//     #pragma unroll
//     for (int remainder = 0; remainder <= pos_dim; remainder++) {
//         // Compute the location of the lattice point explicitly (all but
//         // the last coordinate - it's redundant because they sum to zero)
//         #pragma unroll
//         for (int i = 0; i < pos_dim; i++) {
//             key[i] = static_cast<int>(rem0[i] + remainder);
//             if (rank[i] > pos_dim - remainder)
//                 key[i] -= (pos_dim + 1);
//         }

//         // Retrieve pointer to the value at this vertex.
//         int idx_val=hash_table.retrieve_with_collisions(key);
//         // float *val = const_cast<float *>(hash_table.m_values + idx_val * val_dim );
//         // float *val = const_cast<float *>(hash_table.m_values + idx_val);

//         //store also the splatting indices and weight so that they can be used for the backwards pass
//         if (should_precompute_tensors_for_backward){
//             splatting_indices[idx * (pos_dim + 1) + remainder]=idx_val; //it indexes in m_keys
//             splatting_weights[idx * (pos_dim + 1) + remainder]=barycentric[remainder];
//         }
        

//         //if the vertex exists accumulate its value weighted by the barycentric weight (accumulates also the homogeneous coordinate)
//         // if(idx_val!=-1){
//             #pragma unroll
//             for (int i = 0; i < val_dim ; i++){
//                 // printf("accesing val at position i*nr_positions %d idx val is %d for a total of %d \n", i*nr_positions, idx_val, idx_val+i*nr_positions);
//                 // val_hom[i]+= val[i*nr_positions]* barycentric[remainder];
//                 // values[idx + i*nr_positions]+= val[i*nr_positions]* barycentric[remainder];
//                 // sliced_values[idx][i] += values[idx_val][i] * barycentric[remainder];
//                 val_hom[i] += values[idx_val][i] * barycentric[remainder];
//             }
    
//     }

   

//     // //do not divicde by the homogeneous coordinate, rather just store the value as it is because we will afterwards need the homogeneous coordinate for the backwards passs
//     for (int i = 0; i < val_dim; i++){
//     //         // values[idx*val_dim + i]= val_hom[i] ;
//     //         // values[idx + i*nr_positions]= val_hom[i] ;
//         sliced_values[idx][i]=val_hom[i];
//     }



// }











template<int pos_dim>
__forceinline__ __device__ unsigned int hash(const int *const key) {
    unsigned int k = 0;
    // constexpr uint32_t primes[7] = { 1, 2654435761, 805459861, 3674653429, 2097192037, 1434869437, 2165219737 };  //from https://github.com/NVlabs/tiny-cuda-nn/blob/master/include/tiny-cuda-nn/encodings/grid.h
    #pragma unroll
    for (int i = 0; i < pos_dim; i++) {
        k += key[i];
        k = k * 2531011;
        // k ^=  key[i] * primes[i];
    }
    return k;

    // //do as in voxel hashing: arount line 200 in https://github.com/niessner/VoxelHashing/blob/master/DepthSensingCUDA/Source/VoxelUtilHashSDF.h
    // const int p0 = 73856093;
    // const int p1 = 19349669;
    // const int p2 = 83492791;
    // unsigned int res = ((key[0] * p0) ^ (key[1] * p1) ^ (key[2] * p2));
    // return res;
}

__forceinline__ __device__ int modHash(const unsigned int& n, const int& capacity){
    return(n % capacity);
}

template<int pos_dim>
__forceinline__ __device__ int idx_hash_with_collision(const int * const key, const int& capacity) {

    int h = modHash(hash<pos_dim>(key), capacity);
    // int *e = m_entries + h;
    return h;


}

// //page 86/146 from https://core.ac.uk/download/pdf/85209106.pdf
// __device__ float barycentric_to_c2_continous(const float x) {

//     float y= 6*powf(x,5) - 15*powf(x,4) +10*powf(x,3);

//     return y;
// }
// __device__ float barycentric_to_c1_continous(const float x) {

//     float y= 3*powf(x,2) - 2*powf(x,3);

//     return y;
// }
// __device__ float barycentric_to_c1_continous_back(const float x) {

//     float y= 6*x - 6*powf(x,2);

//     return y;
// }





template<int pos_dim, int val_dim>
__global__ void 
// __launch_bounds__(BLOCK_SIZE) //since the block size is known at compile time we can specify it to the kernel and therefore cuda doesnt need to use heuristics based on code complexity to minimize registry usage
slice_with_collisions_no_precomputation_fast(
    const int nr_positions,
    const int lattice_capacity,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> positions,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> lattice_values,
    const torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> scale_factor,
    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> sliced_values,
    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> splatting_indices,
    torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> splatting_weights,
    const bool should_precompute_tensors_for_backward
    // HashTableGPU hash_table,
    // torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> values
    ) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x; //each thread will deal with a new value


    if(idx>=nr_positions){ //don't go out of bounds
        return;
    }


    
    float elevated[pos_dim + 1];

    //calculate elevated
    // float scaleFactor[pos_dim];
    // float invStdDev = (pos_dim + 1) * sqrt(2.0f / 3);
    // float invStdDev = 1.0;
    // #pragma unroll
    // for (int i = 0; i < pos_dim; i++) {
    //     scaleFactor[i] = 1.0f / (sqrt((float) (i + 1) * (i + 2))) * invStdDev;
    // }
    // embed position vector into the hyperplane
    // first rotate position into the (pd+1)-dimensional hyperplane
    // sm contains the sum of 1..n of our feature vector
    float sm = 0;
    #pragma unroll
    for (int i = pos_dim; i > 0; i--) {
        // float cf = position[i*nr_positions - 1*nr_positions] * scaleFactor[i - 1];
        // float cf = positions[idx][i-1] * scaleFactor[i - 1];
        float cf = positions[idx][i-1] * scale_factor[i - 1];
        // float cf = positions[idx][i-1] ;
        // float cf = position[i - 1] ;
        elevated[i] = sm - i * cf;
        sm += cf;
    }
    elevated[0] = sm;


    int rem0[pos_dim + 1];
    int rank[pos_dim + 1]{0};


   


    // Find the closest 0-colored simplex through rounding
    // greedily search for the closest zero-colored lattice point
    int sum = 0;
    #pragma unroll
    for (int i = 0; i <= pos_dim; i++) {
        float v = elevated[i] * (1.0 / (pos_dim + 1));
        float up = ceil(v) * (pos_dim + 1);
        float down = floor(v) * (pos_dim + 1);
        if (up - elevated[i] < elevated[i] - down) {
            rem0[i] = (int) up;
        } else {
            rem0[i] = (int) down;
        }
        sum += rem0[i];
    }
    sum /= pos_dim + 1;


    // Find the simplex we are in and store it in rank (where rank describes what position coordinate i has in the sorted order of the features values)
    #pragma unroll
    for (int i = 0; i < pos_dim; i++) {
        double di = elevated[i] - rem0[i];
        for (int j = i + 1; j <= pos_dim; j++)
            if (di < elevated[j] - rem0[j])
                rank[i]++;
            else
                rank[j]++;
    }

    // If the point doesn't lie on the plane (sum != 0) bring it back
    #pragma unroll
    for (int i = 0; i <= pos_dim; i++) {
        rank[i] += sum;
        if (rank[i] < 0) {
            rank[i] += pos_dim + 1;
            rem0[i] += pos_dim + 1;
        } else if (rank[i] > pos_dim) {
            rank[i] -= pos_dim + 1;
            rem0[i] -= pos_dim + 1;
        }
    }



    float barycentric[pos_dim + 2]{0};
    // Compute the barycentric coordinates (p.10 in [Adams etal 2010])
    #pragma unroll
    for (int i = 0; i <= pos_dim; i++) {
        float delta = (elevated[i] - rem0[i]) * (1.0 / (pos_dim + 1));
        barycentric[pos_dim - rank[i]] += delta;
        barycentric[pos_dim + 1 - rank[i]] -= delta;
    }
    // Wrap around
    barycentric[0] += 1.0 + barycentric[pos_dim + 1];

    //smoothstep of the linear interpolation like done in the instant ngp paper
    // for (int i = 0; i <= pos_dim; i++) {
        // barycentric[i] = barycentric[i]*barycentric[i] *(3-2*barycentric[i]);
        // float smooth_weight = barycentric[i]*barycentric[i] *(3-2*barycentric[i]);
        // printf("prev weight is %f, new weight is %f \n", barycentric[i], smooth_weight);
        //invere smoothstep https://stackoverflow.com/questions/28740544/inverted-smoothstep
        // if(barycentric[i]>0 && barycentric[i]<1){
            // barycentric[i] = 0.5 - sin(asin(1.0-2.0*barycentric[i])/3.0);
        // }
    // }

    //from higher order barycentric coordinates https://domino.mpi-inf.mpg.de/intranet/ag4/ag4publ.nsf/3b7127147beb1437c125675300686244/637fcbb7f3f5a70fc12573cc00458c99/$FILE/paper.pdf
    //same as instant ngp
    // for (int i = 0; i <= pos_dim; i++) {
        // barycentric[i] = -2* barycentric[i]*barycentric[i] *( barycentric[i] -3.0/2 );
        // barycentric[i] = barycentric[i]*barycentric[i] *(3-2*barycentric[i]);
    // }
    // //renormalize
    // float sum_bar=0.0;
    // for (int i = 0; i <= pos_dim; i++) {
        // sum_bar+=barycentric[i];
    // }
    // printf("sum %f \n", sum_bar);
    // for (int i = 0; i <= pos_dim; i++) {
        // barycentric[i]/=sum_bar;
    // }



    //here we accumulate the values and the homogeneous term
    float val_hom[val_dim]{0};

    int key[pos_dim];
    #pragma unroll
    for (int remainder = 0; remainder <= pos_dim; remainder++) {
        // Compute the location of the lattice point explicitly (all but
        // the last coordinate - it's redundant because they sum to zero)
        #pragma unroll
        for (int i = 0; i < pos_dim; i++) {
            key[i] = static_cast<int>(rem0[i] + remainder);
            if (rank[i] > pos_dim - remainder)
                key[i] -= (pos_dim + 1);
        }

        // Retrieve pointer to the value at this vertex.
        // int idx_val=hash_table.retrieve_with_collisions(key);
        int idx_val=idx_hash_with_collision<pos_dim>(key, lattice_capacity);
        // float *val = const_cast<float *>(hash_table.m_values + idx_val * val_dim );
        // float *val = const_cast<float *>(hash_table.m_values + idx_val);

        //store also the splatting indices and weight so that they can be used for the backwards pass
        if (should_precompute_tensors_for_backward){
            splatting_indices[idx * (pos_dim + 1) + remainder]=idx_val; //it indexes in m_keys
            splatting_weights[idx * (pos_dim + 1) + remainder]=barycentric[remainder];
        }
        

        //if the vertex exists accumulate its value weighted by the barycentric weight (accumulates also the homogeneous coordinate)
        // if(idx_val!=-1){
            #pragma unroll
            for (int i = 0; i < val_dim ; i++){
                // printf("accesing val at position i*nr_positions %d idx val is %d for a total of %d \n", i*nr_positions, idx_val, idx_val+i*nr_positions);
                // val_hom[i]+= val[i*nr_positions]* barycentric[remainder];
                // values[idx + i*nr_positions]+= val[i*nr_positions]* barycentric[remainder];
                // sliced_values[idx][i] += values[idx_val][i] * barycentric[remainder];
                val_hom[i] += lattice_values[idx_val][i] * barycentric[remainder];
            }
    
    }

   

    // //do not divicde by the homogeneous coordinate, rather just store the value as it is because we will afterwards need the homogeneous coordinate for the backwards passs
    #pragma unroll
    for (int i = 0; i < val_dim; i++){
        sliced_values[idx][i]=val_hom[i];
    }



}


template<int pos_dim, int val_dim>
__global__ void 
// __launch_bounds__(BLOCK_SIZE) //since the block size is known at compile time we can specify it to the kernel and therefore cuda doesnt need to use heuristics based on code complexity to minimize registry usage
slice_with_collisions_no_precomputation_fast_mr_loop(
    const int nr_positions,
    const int lattice_capacity,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> positions,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> lattice_values,
    const torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> scale_factor,
    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> sliced_values,
    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> splatting_indices,
    torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> splatting_weights,
    const bool should_precompute_tensors_for_backward
    ) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x; //each thread will deal with a new value


    if(idx>=nr_positions){ //don't go out of bounds
        return;
    }


    
    float elevated[pos_dim + 1];

   
    float sm = 0;
    #pragma unroll
    for (int i = pos_dim; i > 0; i--) {
        float cf = positions[idx][i-1] * scale_factor[i - 1];
        elevated[i] = sm - i * cf;
        sm += cf;
    }
    elevated[0] = sm;


    int rem0[pos_dim + 1];
    int rank[pos_dim + 1]{0};




    // Find the closest 0-colored simplex through rounding
    // greedily search for the closest zero-colored lattice point
    int sum = 0;
    #pragma unroll
    for (int i = 0; i <= pos_dim; i++) {
        float v = elevated[i] * (1.0 / (pos_dim + 1));
        float up = ceil(v) * (pos_dim + 1);
        float down = floor(v) * (pos_dim + 1);
        if (up - elevated[i] < elevated[i] - down) {
            rem0[i] = (int) up;
        } else {
            rem0[i] = (int) down;
        }
        sum += rem0[i];
    }
    sum /= pos_dim + 1;


    // Find the simplex we are in and store it in rank (where rank describes what position coordinate i has in the sorted order of the features values)
    #pragma unroll
    for (int i = 0; i < pos_dim; i++) {
        double di = elevated[i] - rem0[i];
        for (int j = i + 1; j <= pos_dim; j++)
            if (di < elevated[j] - rem0[j])
                rank[i]++;
            else
                rank[j]++;
    }

    // If the point doesn't lie on the plane (sum != 0) bring it back
    #pragma unroll
    for (int i = 0; i <= pos_dim; i++) {
        rank[i] += sum;
        if (rank[i] < 0) {
            rank[i] += pos_dim + 1;
            rem0[i] += pos_dim + 1;
        } else if (rank[i] > pos_dim) {
            rank[i] -= pos_dim + 1;
            rem0[i] -= pos_dim + 1;
        }
    }



    float barycentric[pos_dim + 2]{0};
    // Compute the barycentric coordinates (p.10 in [Adams etal 2010])
    #pragma unroll
    for (int i = 0; i <= pos_dim; i++) {
        float delta = (elevated[i] - rem0[i]) * (1.0 / (pos_dim + 1));
        barycentric[pos_dim - rank[i]] += delta;
        barycentric[pos_dim + 1 - rank[i]] -= delta;
    }
    // Wrap around
    barycentric[0] += 1.0 + barycentric[pos_dim + 1];


    //here we accumulate the values and the homogeneous term
    float val_hom[val_dim]{0};

    int key[pos_dim];
    #pragma unroll
    for (int remainder = 0; remainder <= pos_dim; remainder++) {
        // Compute the location of the lattice point explicitly (all but
        // the last coordinate - it's redundant because they sum to zero)
        #pragma unroll
        for (int i = 0; i < pos_dim; i++) {
            key[i] = static_cast<int>(rem0[i] + remainder);
            if (rank[i] > pos_dim - remainder)
                key[i] -= (pos_dim + 1);
        }

        // Retrieve pointer to the value at this vertex.
        int idx_val=idx_hash_with_collision<pos_dim>(key, lattice_capacity);

        //store also the splatting indices and weight so that they can be used for the backwards pass
        if (should_precompute_tensors_for_backward){
            splatting_indices[idx * (pos_dim + 1) + remainder]=idx_val; //it indexes in m_keys
            splatting_weights[idx * (pos_dim + 1) + remainder]=barycentric[remainder];
        }
        

        //if the vertex exists accumulate its value weighted by the barycentric weight (accumulates also the homogeneous coordinate)
            #pragma unroll
            for (int i = 0; i < val_dim ; i++){
                val_hom[i] += lattice_values[idx_val][i] * barycentric[remainder];
            }
    
    }

   

    // //do not divicde by the homogeneous coordinate, rather just store the value as it is because we will afterwards need the homogeneous coordinate for the backwards passs
    #pragma unroll
    for (int i = 0; i < val_dim; i++){
        sliced_values[idx][i]=val_hom[i];
        // sliced_values[i][idx]=val_hom[i];
    }



}


// __constant__ float scalings_constants[14*3];

template<int pos_dim, int val_dim>
__global__ void 
// __launch_bounds__(BLOCK_SIZE) //since the block size is known at compile time we can specify it to the kernel and therefore cuda doesnt need to use heuristics based on code complexity to minimize registry usage
slice_with_collisions_no_precomputation_fast_mr_monolithic_no_accessors(
    const int nr_positions,
    const int lattice_capacity,
    const float* positions,
    float* lattice_values_monolithic,
    const float* scale_factor,
    const float* random_shift_monolithic,
    const float* anneal_window,
    float* sliced_values_monolithic,
    // torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> elevated_out,
    int* splatting_indices,
    at::Half* splatting_weights,
    const bool require_lattice_values_grad,
    const bool require_positions_grad
    ) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x; //each thread will deal with a new value


    if(idx>=nr_positions){ //don't go out of bounds
        return;
    }

    const uint32_t level = blockIdx.y; // <- the level is the same for all threads
    // const uint32_t level = 1; // <- the level is the same for all threads


    
    float elevated[pos_dim + 1];




    float sm = 0;
    #pragma unroll
    for (int i = pos_dim; i > 0; i--) {
        // float cf = (positions[idx][i-1] +random_shift_monolithic[level][i-1]  ) * scale_factor[level][i - 1];

        float cf = (positions[idx*3+i-1] +random_shift_monolithic[level*3+i-1]  ) * scale_factor[level*3+i-1];
        
        elevated[i] = sm - i * cf;
        sm += cf;
    }
    elevated[0] = sm;


    int rem0[pos_dim + 1];
    int rank[pos_dim + 1]{0};




    // Find the closest 0-colored simplex through rounding
    // greedily search for the closest zero-colored lattice point
    int sum = 0;
    #pragma unroll
    for (int i = 0; i <= pos_dim; i++) {
        float v = elevated[i] * (1.0 / (pos_dim + 1));
        float up = ceil(v) * (pos_dim + 1);
        float down = floor(v) * (pos_dim + 1);
        if (up - elevated[i] < elevated[i] - down) {
            rem0[i] = (int) up;
        } else {
            rem0[i] = (int) down;
        }
        sum += rem0[i];
    }
    sum /= pos_dim + 1;


    // Find the simplex we are in and store it in rank (where rank describes what position coordinate i has in the sorted order of the features values)
    #pragma unroll
    for (int i = 0; i < pos_dim; i++) {
        double di = elevated[i] - rem0[i];
        for (int j = i + 1; j <= pos_dim; j++)
            if (di < elevated[j] - rem0[j])
                rank[i]++;
            else
                rank[j]++;
    }

    // If the point doesn't lie on the plane (sum != 0) bring it back
    #pragma unroll
    for (int i = 0; i <= pos_dim; i++) {
        rank[i] += sum;
        if (rank[i] < 0) {
            rank[i] += pos_dim + 1;
            rem0[i] += pos_dim + 1;
        } else if (rank[i] > pos_dim) {
            rank[i] -= pos_dim + 1;
            rem0[i] -= pos_dim + 1;
        }
    }



    float barycentric[pos_dim + 2]{0};
    // Compute the barycentric coordinates (p.10 in [Adams etal 2010])
    #pragma unroll
    for (int i = 0; i <= pos_dim; i++) {
        float delta = (elevated[i] - rem0[i]) * (1.0 / (pos_dim + 1));
        barycentric[pos_dim - rank[i]] += delta;
        barycentric[pos_dim + 1 - rank[i]] -= delta;
    }
    // Wrap around
    barycentric[0] += 1.0 + barycentric[pos_dim + 1];



    //smoothstep of the linear interpolation like done in the instant ngp paper
    // for (int i = 0; i <= pos_dim; i++) {
        // barycentric[i] = barycentric[i]*barycentric[i] *(3-2*barycentric[i]);
        // float smooth_weight = barycentric[i]*barycentric[i] *(3-2*barycentric[i]);
        // printf("prev weight is %f, new weight is %f \n", barycentric[i], smooth_weight);
        //invere smoothstep https://stackoverflow.com/questions/28740544/inverted-smoothstep
        // if(barycentric[i]>0 && barycentric[i]<1){
            // barycentric[i] = 0.5 - sin(asin(1.0-2.0*barycentric[i])/3.0);
        // }
    // }

    //from higher order barycentric coordinates https://domino.mpi-inf.mpg.de/intranet/ag4/ag4publ.nsf/3b7127147beb1437c125675300686244/637fcbb7f3f5a70fc12573cc00458c99/$FILE/paper.pdf
    //same as instant ngp
    // for (int i = 0; i <= pos_dim; i++) {
        // barycentric[i] = -2* barycentric[i]*barycentric[i] *( barycentric[i] -3.0/2 );
        // barycentric[i] = barycentric[i]*barycentric[i] *(3-2*barycentric[i]);
    // }
    // //renormalize
    // float sum_bar=0.0;
    // for (int i = 0; i <= pos_dim; i++) {
        // sum_bar+=barycentric[i];
    // }
    // printf("sum %f \n", sum_bar);
    // for (int i = 0; i <= pos_dim; i++) {
        // barycentric[i]/=sum_bar;
    // }


    //attempt 2 like in page 83/146 https://core.ac.uk/download/pdf/85209106.pdf
    // for (int i = 0; i <= pos_dim; i++) {
    //     barycentric[i]=barycentric_to_c2_continous(barycentric[i]);
    // }
    // //renormalize
    // float sum_bar=0.0;
    // for (int i = 0; i <= pos_dim; i++) {
    //     sum_bar+=barycentric[i];
    // }
    // // printf("sum %f \n", sum_bar);
    // for (int i = 0; i <= pos_dim; i++) {
    //     barycentric[i]/=sum_bar;
    // }


    





    //here we accumulate the values and the homogeneous term
    // float val_hom[val_dim]{0};
    float2 val_hom;
    val_hom.x=0;
    val_hom.y=0;

    float w_lvl= anneal_window[level];

    int key[pos_dim];
    #pragma unroll
    for (int remainder = 0; remainder <= pos_dim; remainder++) {
        // Compute the location of the lattice point explicitly (all but
        // the last coordinate - it's redundant because they sum to zero)
        #pragma unroll
        for (int i = 0; i < pos_dim; i++) {
            key[i] = static_cast<int>(rem0[i] + remainder);
            if (rank[i] > pos_dim - remainder)
                key[i] -= (pos_dim + 1);
        }

        // Retrieve pointer to the value at this vertex.
        int idx_val=idx_hash_with_collision<pos_dim>(key, lattice_capacity);

        //store also the splatting indices and weight so that they can be used for the backwards pass
        if (require_lattice_values_grad || require_positions_grad){
            // splatting_indices[level][remainder][idx]=idx_val;
            // splatting_weights[level][remainder][idx]=barycentric[remainder] * w_lvl; //we save the barycentric with the anneal window so in the backward pass everything is masked correct by the annealed mask

             splatting_indices[level*nr_positions*4 + remainder*nr_positions+idx]=idx_val;
             splatting_weights[level*nr_positions*4 + remainder*nr_positions+idx]=barycentric[remainder] * w_lvl;

        }
        

        //if the vertex exists accumulate its value weighted by the barycentric weight (accumulates also the homogeneous coordinate)
            // #pragma unroll
            float w= barycentric[remainder] * w_lvl;
            // for (int i = 0; i < val_dim ; i++){
                // val_hom[i] += lattice_values_monolithic[level][idx_val][i] * w;


                // float* fv=&lattice_values_monolithic[level][idx_val][0];
                // float2 new_val=reinterpret_cast<float2*>( fv )[0];
                // val_hom.x += val_hom.x + new_val.x*w;
                // val_hom.y += val_hom.y + new_val.y*w;


                //get ptr to this value of the vertex
                float* fv=lattice_values_monolithic + (level*2*lattice_capacity+idx_val*2);
                float2 new_val=reinterpret_cast<float2*>( fv )[0];
                val_hom.x += val_hom.x + new_val.x*w;
                val_hom.y += val_hom.y + new_val.y*w;


    
    }

   

    // //do not divicde by the homogeneous coordinate, rather just store the value as it is because we will afterwards need the homogeneous coordinate for the backwards passs
    // #pragma unroll
    // for (int i = 0; i < val_dim; i++){

        // sliced_values_monolithic[level][i][idx]=val_hom[i]; //fastest
    // }
    // reinterpret_cast<float2*>(sliced_values_monolithic[level][idx])=val_hom;

    // sliced_values_monolithic[level][0][idx]=val_hom.x;
    // sliced_values_monolithic[level][1][idx]=val_hom.y;

    //store------------
    sliced_values_monolithic[level*nr_positions*2 + 0*nr_positions + idx]=val_hom.x;
    sliced_values_monolithic[level*nr_positions*2 + 1*nr_positions + idx]=val_hom.y;


    //store but the value is the dense one
    // sliced_values_monolithic[level][idx][0]=val_hom.x;
    // sliced_values_monolithic[level][idx][1]=val_hom.y;
    // sliced_values_monolithic[level*nr_positions*2 + idx*2+0]=val_hom.x;
    // sliced_values_monolithic[level*nr_positions*2 + idx*2+1]=val_hom.y;
    // float2* val_to_store_ptr= reinterpret_cast<float2*>( sliced_values_monolithic +(level*nr_positions*2 + idx*2));
    // val_to_store_ptr[0]=val_hom;



    



}


template<int pos_dim, int val_dim>
__global__ void 
__launch_bounds__(BLOCK_SIZE) //since the block size is known at compile time we can specify it to the kernel and therefore cuda doesnt need to use heuristics based on code complexity to minimize registry usage
slice_with_collisions_no_precomputation_fast_mr_monolithic(
    const int nr_positions,
    const int lattice_capacity,
    const int nr_resolutions,
    const int nr_resolutions_extra,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> positions,
    #if LATTICE_HALF_PRECISION
        torch::PackedTensorAccessor32<at::Half,3,torch::RestrictPtrTraits> lattice_values_monolithic,
    #else
        torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> lattice_values_monolithic,
    #endif
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> scale_factor,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> random_shift_monolithic,
    const torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> anneal_window,
    // #if LATTICE_HALF_PRECISION
        // torch::PackedTensorAccessor32<at::Half,3,torch::RestrictPtrTraits> sliced_values_monolithic,
    // #else
        torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> sliced_values_monolithic,
    // #endif
    // torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> elevated_out,
    // torch::PackedTensorAccessor32<int,3,torch::RestrictPtrTraits> splatting_indices,
    // #if LATTICE_HALF_PRECISION
    //     torch::PackedTensorAccessor32<at::Half,3,torch::RestrictPtrTraits> splatting_weights,
    // #else 
    //     torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> splatting_weights,
    // #endif
    const bool concat_points, 
    const float points_scaling,
    const bool require_lattice_values_grad,
    const bool require_positions_grad
    ) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x; //each thread will deal with a new value


    if(idx>=nr_positions){ //don't go out of bounds
        return;
    }

    const uint32_t level = blockIdx.y; // <- the level is the same for all threads
    // const uint32_t level = 1; // <- the level is the same for all threads


    //if we are in one of the extra resolutions and we are also concating the points, then do so
    int idx_extra_level=level-nr_resolutions;
    if (idx_extra_level>=0){ //we are in one of the extra levels
        //check if this extra level that we are in is within the bound of the pos_dim
        //we can have for example 2 extra levels with 2 val dim each, so a total of 4 more dimensions. But our pos dim =3 so the last val dim is just 0

        // printf("adding extra dimensions\n");

        for (int i = 0; i < val_dim; i++){
            int position_dimension_to_copy_from=i+idx_extra_level*val_dim; //for the first resolution this is 0 and 1 , for the other resolution it will be 2 and 3.
            if(position_dimension_to_copy_from<pos_dim){
                // printf("copying from dimensions %d\n", position_dimension_to_copy_from);
                sliced_values_monolithic[level][i][idx]=positions[idx][position_dimension_to_copy_from] * points_scaling;
            }else{ //we are in the 4 dimensions but we have only posdim=3 so we just put a zero here
                sliced_values_monolithic[level][i][idx]=0.0;
                // printf("putting 0 at dimension %d\n", i);
            }
        }



        return;

    }


    
    float elevated[pos_dim + 1];


    //load the scale factor into shared memory
    // __shared__ float scale_factor_shared[pos_dim];
    // if(threadIdx.x==0){
    //         for (int sp=0; sp<pos_dim; sp++) {
    //             scale_factor_shared[sp]=scale_factor[level][sp];
    //         }
    // }
    // __syncthreads();

    // //get position vectorized load
    // float* pos_ptr=&positions[idx][0];
    // float3 pos=reinterpret_cast<float3*>( pos_ptr )[0];
    // //attempt 2
    // float sm = 0;
    // float cf=0;
    // //i=3
    // cf = (pos.z +random_shift_monolithic[level][2]  ) * scale_factor[level][2];
    // elevated[3] = sm - 3 * cf;
    // sm += cf;
    // //i=2
    // cf = (pos.y +random_shift_monolithic[level][1]  ) * scale_factor[level][1];
    // elevated[2] = sm - 2 * cf;
    // sm += cf;
    // //i=1
    // cf = (pos.x +random_shift_monolithic[level][0]  ) * scale_factor[level][0];
    // elevated[1] = sm - 1 * cf;
    // sm += cf;
    // //
    // elevated[0] = sm;


    float sm = 0;
    #pragma unroll
    for (int i = pos_dim; i > 0; i--) {
        float cf = (positions[idx][i-1] +random_shift_monolithic[level][i-1]  ) * scale_factor[level][i - 1];
        // float cf = positions[idx][i-1] * scale_factor[level][i - 1];
        // printf("sf is  %f sf_shared %f  \n", scale_factor[level][i - 1],   scale_factor_shared[i - 1] );
        // float cf = positions[idx][i-1] * scale_factor_shared[i - 1];
        // float cf = positions[idx][i-1] ;
        // float cf = positions[idx][i-1] * scalings_constants[(i - 1)  + level*3];
        // float cf = positions[idx][i-1] * scalings[(i - 1)  + level*3];
        // float cf = positions[i-1][idx] * scale_factor[level][i - 1];
        elevated[i] = sm - i * cf;
        sm += cf;
    }
    elevated[0] = sm;

    // constexpr int posdim1_pow2=upper_power_of_two(pos_dim+1);
    // constexpr int posdim2_pow2=upper_power_of_two(pos_dim+2);
    int rem0[pos_dim+1];
    int rank[pos_dim+1]{0};




    // Find the closest 0-colored simplex through rounding
    // greedily search for the closest zero-colored lattice point
    int sum = 0;
    #pragma unroll
    for (int i = 0; i <= pos_dim; i++) {
        float v = elevated[i] * (1.0 / (pos_dim + 1));
        float up = ceil(v) * (pos_dim + 1);
        float down = floor(v) * (pos_dim + 1);
        if (up - elevated[i] < elevated[i] - down) {
            rem0[i] = (int) up;
        } else {
            rem0[i] = (int) down;
        }
        sum += rem0[i];
    }
    sum /= pos_dim + 1;


    // Find the simplex we are in and store it in rank (where rank describes what position coordinate i has in the sorted order of the features values)
    #pragma unroll
    for (int i = 0; i < pos_dim; i++) {
        double di = elevated[i] - rem0[i];
        for (int j = i + 1; j <= pos_dim; j++)
            if (di < elevated[j] - rem0[j])
                rank[i]++;
            else
                rank[j]++;
    }

    // If the point doesn't lie on the plane (sum != 0) bring it back
    #pragma unroll
    for (int i = 0; i <= pos_dim; i++) {
        rank[i] += sum;
        if (rank[i] < 0) {
            rank[i] += pos_dim + 1;
            rem0[i] += pos_dim + 1;
        } else if (rank[i] > pos_dim) {
            rank[i] -= pos_dim + 1;
            rem0[i] -= pos_dim + 1;
        }
    }



    float barycentric[pos_dim + 2]{0};
    // Compute the barycentric coordinates (p.10 in [Adams etal 2010])
    #pragma unroll
    for (int i = 0; i <= pos_dim; i++) {
        float delta = (elevated[i] - rem0[i]) * (1.0 / (pos_dim + 1));
        barycentric[pos_dim - rank[i]] += delta;
        barycentric[pos_dim + 1 - rank[i]] -= delta;
    }
    // Wrap around
    barycentric[0] += 1.0 + barycentric[pos_dim + 1];



    //smoothstep of the linear interpolation like done in the instant ngp paper
    // for (int i = 0; i <= pos_dim; i++) {
        // barycentric[i] = barycentric[i]*barycentric[i] *(3-2*barycentric[i]);
        // float smooth_weight = barycentric[i]*barycentric[i] *(3-2*barycentric[i]);
        // printf("prev weight is %f, new weight is %f \n", barycentric[i], smooth_weight);
        //invere smoothstep https://stackoverflow.com/questions/28740544/inverted-smoothstep
        // if(barycentric[i]>0 && barycentric[i]<1){
            // barycentric[i] = 0.5 - sin(asin(1.0-2.0*barycentric[i])/3.0);
        // }
    // }

    //from higher order barycentric coordinates https://domino.mpi-inf.mpg.de/intranet/ag4/ag4publ.nsf/3b7127147beb1437c125675300686244/637fcbb7f3f5a70fc12573cc00458c99/$FILE/paper.pdf
    //same as instant ngp
    // for (int i = 0; i <= pos_dim; i++) {
        // barycentric[i] = -2* barycentric[i]*barycentric[i] *( barycentric[i] -3.0/2 );
        // barycentric[i] = barycentric[i]*barycentric[i] *(3-2*barycentric[i]);
    // }
    // //renormalize
    // float sum_bar=0.0;
    // for (int i = 0; i <= pos_dim; i++) {
        // sum_bar+=barycentric[i];
    // }
    // printf("sum %f \n", sum_bar);
    // for (int i = 0; i <= pos_dim; i++) {
        // barycentric[i]/=sum_bar;
    // }


    //attempt 2 like in page 86/146 https://core.ac.uk/download/pdf/85209106.pdf
    // for (int i = 0; i <= pos_dim; i++) {
    //     barycentric[i]=barycentric_to_c1_continous(barycentric[i]);
    // }
    // //renormalize
    // float sum_bar=0.0;
    // for (int i = 0; i <= pos_dim; i++) {
    //     sum_bar+=barycentric[i];
    // }
    // // printf("sum %f \n", sum_bar);
    // for (int i = 0; i <= pos_dim; i++) {
    //     barycentric[i]/=sum_bar;
    // }


    





    //here we accumulate the values and the homogeneous term
    // float val_hom[val_dim]{0};
    #if LATTICE_HALF_PRECISION
        // __nv_bfloat162 val_hom_vec;
        // val_hom_vec.x=__float2bfloat16(0);
        // val_hom_vec.y=__float2bfloat16(0);
        float2 val_hom_vec;
        val_hom_vec.x=0;
        val_hom_vec.y=0;
    #else
        float2 val_hom_vec;
        val_hom_vec.x=0;
        val_hom_vec.y=0;
    #endif

    float w_lvl= anneal_window[level];

    int key[pos_dim];
    #pragma unroll
    for (int remainder = 0; remainder <= pos_dim; remainder++) {
        // Compute the location of the lattice point explicitly (all but
        // the last coordinate - it's redundant because they sum to zero)
        #pragma unroll
        for (int i = 0; i < pos_dim; i++) {
            key[i] =rem0[i] + remainder;
            if (rank[i] > pos_dim - remainder)
                key[i] -= (pos_dim + 1);
        }

        // Retrieve pointer to the value at this vertex.
        int idx_val=idx_hash_with_collision<pos_dim>(key, lattice_capacity);

        //store also the splatting indices and weight so that they can be used for the backwards pass
        // if (require_lattice_values_grad || require_positions_grad){
        //     // splatting_indices[level][remainder][idx]=idx_val;
        //     // splatting_weights[level][remainder][idx]=barycentric[remainder] * w_lvl; //we save the barycentric with the anneal window so in the backward pass everything is masked correct by the annealed mask

        //     //tranposed
        //     splatting_indices[level][idx][remainder]=idx_val;
        //     splatting_weights[level][idx][remainder]=barycentric[remainder] * w_lvl;
        // }
        

        //if the vertex exists accumulate its value weighted by the barycentric weight (accumulates also the homogeneous coordinate)
        float w= barycentric[remainder] * w_lvl;

        // #pragma unroll
        // for (int i = 0; i < val_dim ; i++){
            // val_hom[i] += lattice_values_monolithic[level][idx_val][i] * w;
        // }

        //vectorized loads 
        #if LATTICE_HALF_PRECISION
            // __nv_bfloat16* fv=static_cast<__nv_bfloat16*> (  (void*)&lattice_values_monolithic[level][idx_val][0]); 
            // __nv_bfloat162 new_val=reinterpret_cast<__nv_bfloat162*>( fv )[0];
            // //srote as bfloat16
            // // val_hom_vec.x = val_hom_vec.x + new_val.x*  	__float2bfloat16(w);
            // // val_hom_vec.y = val_hom_vec.y + new_val.y*      __float2bfloat16(w);
            // //store as float
            // val_hom_vec.x = val_hom_vec.x + __bfloat162float(new_val.x)*  w;
            // val_hom_vec.y = val_hom_vec.y + __bfloat162float(new_val.y)*  w;

            //do it with float16
            __half* fv=static_cast<__half*> (  (void*)&lattice_values_monolithic[level][idx_val][0]); 
            __half2 new_val=reinterpret_cast<__half2*>( fv )[0];
            val_hom_vec.x = val_hom_vec.x + __half2float(new_val.x)*  w;
            val_hom_vec.y = val_hom_vec.y + __half2float(new_val.y)*  w;
        #else
            float* fv=&lattice_values_monolithic[level][idx_val][0];
            float2 new_val=reinterpret_cast<float2*>( fv )[0];
            val_hom_vec.x = val_hom_vec.x + new_val.x*w;
            val_hom_vec.y = val_hom_vec.y + new_val.y*w;
        #endif

        //compare
        // float* fv=&lattice_values_monolithic[level][idx_val][0];
        // float2 new_val=reinterpret_cast<float2*>( fv )[0];
        // if (lattice_values_monolithic[level][idx_val][0]!=new_val.x){
        //     printf("x doesnt match\n");
        // }
        // if (lattice_values_monolithic[level][idx_val][1]!=new_val.y){
        //     printf("x doesnt match\n");
        // }
    }

   

    // //do not divicde by the homogeneous coordinate, rather just store the value as it is because we will afterwards need the homogeneous coordinate for the backwards passs
    // #pragma unroll
    // for (int i = 0; i < val_dim; i++){
        // sliced_values_monolithic[level][i][idx]=val_hom[i]; //fastest
    // }


    // if (val_hom_vec.x!=val_hom[0]){
    //     printf("x doesnt match\n");
    // }
    // if (val_hom_vec.y!=val_hom[1]){
    //     printf("y doesnt match\n");
    // }


    // vectorized stores
    // #if LATTICE_HALF_PRECISION
        // sliced_values_monolithic[level][0][idx]=__float2bfloat16(val_hom_vec.x);
        // sliced_values_monolithic[level][1][idx]=__float2bfloat16(val_hom_vec.y);

        //do it with half
        // sliced_values_monolithic[level][0][idx]=__float2half(val_hom_vec.x);
        // sliced_values_monolithic[level][1][idx]=__float2half(val_hom_vec.y);

        //do it with half transposed
        // sliced_values_monolithic[level][idx][0]=__float2half(val_hom_vec.x);
        // sliced_values_monolithic[level][idx][1]=__float2half(val_hom_vec.y);

        //do it with half tanposed but vectorized
        // __half2 val_hom_half2=__float22half2_rn(val_hom_vec);
        // __half2* ptr_dest=static_cast<__half2*> (  (void*)&sliced_values_monolithic[level][idx][0]);
        // *ptr_dest=val_hom_half2;

    // #else
        sliced_values_monolithic[level][0][idx]=val_hom_vec.x;
        sliced_values_monolithic[level][1][idx]=val_hom_vec.y;
    // #endif
    // sliced_values_monolithic[level][0][idx]=val_hom_vec.x;
    // sliced_values_monolithic[level][1][idx]=val_hom_vec.y;

    //vectorized stores tranposed
    // sliced_values_monolithic[level][idx][0]=val_hom_vec.x;
    // sliced_values_monolithic[level][idx][1]=val_hom_vec.y;


    



}










struct __device_builtin__ __builtin_align__(8) __half4
{
    __half x, y, z, w;
};




template<int pos_dim>
__global__ void 
__launch_bounds__(BLOCK_SIZE) //since the block size is known at compile time we can specify it to the kernel and therefore cuda doesnt need to use heuristics based on code complexity to minimize registry usage
slice_backwards_with_precomputation_no_homogeneous(
    const int nr_positions,
    const int val_dim, 
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> grad_sliced_values,
    const torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> splatting_indices,
    const torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> splatting_weights,
    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> val_out
    ){


    //values_vertices refers to the values that the lattice had in the forward pass. it has size m_hash_table_capcity x (val_dim+1)
    //grad_sliced_values is the gradient of the loss with respect to the sliced out values which has size nr_positions x val_dim
    const int idx = blockIdx.x * blockDim.x + threadIdx.x; // each thread will deal with one position
    if(idx >= nr_positions){
        return;
    }

    
    
    for(int color=0; color<pos_dim+1; color++){
        // int index_into_m_entries=round(splatting_indices_and_weights[ idx * (pos_dim + 1)*2 + color*2 + 0]);
        int splatting_idx = splatting_indices[ idx * (pos_dim + 1) + color];
        if(splatting_idx>=0){

            float weight = splatting_weights[ idx * (pos_dim + 1) + color];
            

            //acumulate the values
            for (int j = 0; j < val_dim; j++) {
                // #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 200)
                    // printf("accumulating weighted gradient %f where gradient is %f and weight is %f  \n", grad_sliced_val[j]*weight, grad_sliced_val[j], weight );
                    // #warning CUDA ARCH IS FINE
                    atomicAdd(&val_out[splatting_idx][j], grad_sliced_values[idx][j]*weight);
                // #else 
                    // #warning CUDA ARCH NEEDS TO BE AT LEAST 200 IN ORDER TO ENABLE ATOMIC OPERATIONS!
                // #endif
            }

        
        }else{
            // printf("splatting idx is not valid for idx %d\n", idx );
        }


    }


}

template<int pos_dim, int val_dim>
__global__ void 
__launch_bounds__(BLOCK_SIZE_BACK) //since the block size is known at compile time we can specify it to the kernel and therefore cuda doesnt need to use heuristics based on code complexity to minimize registry usage
slice_backwards_with_precomputation_no_homogeneous_mr_monolithic(
    const int nr_positions,
    #if LATTICE_HALF_PRECISION
        const torch::PackedTensorAccessor32<at::Half,3,torch::RestrictPtrTraits> grad_sliced_values_monolithic,
    #else
        const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> grad_sliced_values_monolithic,
    #endif
    torch::PackedTensorAccessor32<int,3,torch::RestrictPtrTraits> splatting_indices,
    // torch::PackedTensorAccessor32<at::Half,3,torch::RestrictPtrTraits> splatting_weights,
    // torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> splatting_weights,
    #if LATTICE_HALF_PRECISION
        torch::PackedTensorAccessor32<at::Half,3,torch::RestrictPtrTraits> splatting_weights,
    #else 
        torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> splatting_weights,
    #endif
    #if LATTICE_HALF_PRECISION 
        torch::PackedTensorAccessor32<at::Half,3,torch::RestrictPtrTraits> lattice_values_monolithic
    #else 
        torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> lattice_values_monolithic
    #endif
    ){


    //values_vertices refers to the values that the lattice had in the forward pass. it has size m_hash_table_capcity x (val_dim+1)
    //grad_sliced_values is the gradient of the loss with respect to the sliced out values which has size nr_positions x val_dim
    const int idx = blockIdx.x * blockDim.x + threadIdx.x; // each thread will deal with one position
    if(idx >= nr_positions){
        return;
    }

    const uint32_t level = blockIdx.y; // <- the level is the same for all threads


    // // do it with vectorized loads and stores
    // //load indices
    // int* indices_ptr=&splatting_indices[level][idx][0];
    // int4 indices_vec=reinterpret_cast<int4*>( indices_ptr )[0];
    // //load weights
    // float* weights_ptr=reinterpret_cast<float*>( &splatting_weights[level][idx][0]);
    // float4 weights_vec=reinterpret_cast<float4*>( weights_ptr )[0];


    // //get the values of this position
    // float2 val_at_position;
    // val_at_position.x=grad_sliced_values_monolithic[level][0][idx];
    // val_at_position.y=grad_sliced_values_monolithic[level][1][idx];
    // //tranposed
    // // float* grad_val_ptr=&grad_sliced_values_monolithic[level][idx][0];
    // // float2 val_at_position=reinterpret_cast<float2*>( grad_val_ptr )[0];

    // //splat the 2 values onto the 4 vertices of the simplex
    // //v0
    // atomicAdd(&lattice_values_monolithic[level][0][indices_vec.x],   val_at_position.x* weights_vec.x  );
    // atomicAdd(&lattice_values_monolithic[level][1][indices_vec.x],   val_at_position.y* weights_vec.x  );
    // //v1
    // atomicAdd(&lattice_values_monolithic[level][0][indices_vec.y],   val_at_position.x* weights_vec.y  );
    // atomicAdd(&lattice_values_monolithic[level][1][indices_vec.y],   val_at_position.y* weights_vec.y  );
    // //v2
    // atomicAdd(&lattice_values_monolithic[level][0][indices_vec.z],   val_at_position.x* weights_vec.z  );
    // atomicAdd(&lattice_values_monolithic[level][1][indices_vec.z],   val_at_position.y* weights_vec.z  );
    // //v3
    // atomicAdd(&lattice_values_monolithic[level][0][indices_vec.w],   val_at_position.x* weights_vec.w  );
    // atomicAdd(&lattice_values_monolithic[level][1][indices_vec.w],   val_at_position.y* weights_vec.w  );


    

    

    //DO NOT USE this one becasue it accessed grad_sliced_values way more than necessary    
    // #pragma unroll 
    // for(int color=0; color<pos_dim+1; color++){
    //     // int index_into_m_entries=round(splatting_indices_and_weights[ idx * (pos_dim + 1)*2 + color*2 + 0]);
    //     // int splatting_idx = splatting_indices[ idx * (pos_dim + 1) + color];

    //     // int splatting_idx = splatting_indices[level][color][idx];
    //     int splatting_idx = splatting_indices[level][idx][color];

    //     if(splatting_idx>=0){

    //         // float weight = splatting_weights[level][color][idx];
    //         float weight = splatting_weights[level][idx][color];
            

    //         //acumulate the values
    //         #pragma unroll
    //         for (int j = 0; j < val_dim; j++) {

    //             float weighted_grad=grad_sliced_values_monolithic[level][j][idx]*weight;
    //             // float weighted_grad=grad_sliced_values_monolithic[level][idx][j]*weight;

    //             // if(fabs(weighted_grad)>1e-15){
    //                 // atomicAdd(&lattice_values_monolithic[level][splatting_idx][j], weighted_grad  );
    //                 atomicAdd(&lattice_values_monolithic[level][j][splatting_idx], weighted_grad  ); //fastest
    //                 // lattice_values_monolithic[level][j][splatting_idx]=weighted_grad;
    //                 // #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 200)
    //                 // #else 
    //                 //     #warning CUDA ARCH NEEDS TO BE AT LEAST 200 IN ORDER TO ENABLE ATOMIC OPERATIONS!
    //                 // #endif
    //             // }
              
              
    //         }

        
    //     }


    // }




    //attempt 2
    //get the value at the position
    #if LATTICE_HALF_PRECISION 
        float grad_pos[val_dim];
        #pragma unroll
        for (int j = 0; j < val_dim; j++) {
            grad_pos[j]=__half2float(grad_sliced_values_monolithic[level][j][idx]);
        }
    #else 
        float grad_pos[val_dim];
        #pragma unroll
        for (int j = 0; j < val_dim; j++) {
            grad_pos[j]=grad_sliced_values_monolithic[level][j][idx];
        }
    #endif


    // int* indices_ptr=&splatting_indices[level][idx][0];
    // int4 indices_vec=reinterpret_cast<int4*>( indices_ptr )[0];
    // //load weights
    // #if LATTICE_HALF_PRECISION 
    //     __half* weights_ptr=reinterpret_cast<__half*>( &splatting_weights[level][idx][0]);
    //     __half4 weights_vec=reinterpret_cast<__half4*>( weights_ptr )[0];
    // #else 
    //     float* weights_ptr=reinterpret_cast<float*>( &splatting_weights[level][idx][0]);
    //     float4 weights_vec=reinterpret_cast<float4*>( weights_ptr )[0];
    // #endif



    #pragma unroll 
    for(int color=0; color<pos_dim+1; color++){

        // int splatting_idx = splatting_indices[level][color][idx];
        int splatting_idx = splatting_indices[level][idx][color];
        // int splatting_idx;
        // switch(color){
        //     case(0): splatting_idx=indices_vec.x;break;
        //     case(1): splatting_idx=indices_vec.y;break;
        //     case(2): splatting_idx=indices_vec.z;break;
        //     case(3): splatting_idx=indices_vec.w;break;
        // }


        // float weight = splatting_weights[level][color][idx];
        float weight = splatting_weights[level][idx][color];
        // float weight;
        // switch(color){
        //     case(0): weight=weights_vec.x;break;
        //     case(1): weight=weights_vec.y;break;
        //     case(2): weight=weights_vec.z;break;
        //     case(3): weight=weights_vec.w;break;
        // }


        //do it with only one half2 accumulate 
        #if LATTICE_HALF_PRECISION 
            __half2 weighted_grad;
            weighted_grad.x=__float2half(grad_pos[0]* weight);
            weighted_grad.y=__float2half(grad_pos[1]* weight);
            __half2* ptr= static_cast<__half2*>((void*)&lattice_values_monolithic[level][splatting_idx][0]);
            atomicAdd(ptr, weighted_grad  ); 
        #else 



            //acumulate the values
            #pragma unroll
            for (int j = 0; j < val_dim; j++) {

                // #if LATTICE_HALF_PRECISION 
                //     // const __nv_bfloat16 weighted_grad=grad_pos[j]* __float2bfloat16 (weight);
                //     // const __nv_bfloat16* ptr= static_cast<__nv_bfloat16*>((void*)&lattice_values_monolithic[level][j][splatting_idx]);
                //     // atomicAdd(ptr, weighted_grad  ); //REQUIRES compute capability >8.0 so RTX 3000 series

                //     //do it with half DO NOT DO IT LIKE THISBECAUSE adotmic add for half is super slow
                //     __half weighted_grad=__float2half(grad_pos[j]* weight);
                //     __half* ptr= static_cast<__half*>((void*)&lattice_values_monolithic[level][j][splatting_idx]);
                //     atomicAdd(ptr, weighted_grad  ); 
                // #else 
                    float weighted_grad=grad_pos[j]*weight;
                    // atomicAdd(&lattice_values_monolithic[level][j][splatting_idx], weighted_grad  ); //fastest
                    atomicAdd(&lattice_values_monolithic[level][splatting_idx][j], weighted_grad  ); //fastest
                // #endif
                
            }
        
        #endif

    } 





    // //attemp to make it even faster by vectoried loads
    // //load indices
    // int* indices_ptr=&splatting_indices[level][idx][0];
    // int4 indices_vec=reinterpret_cast<int4*>( indices_ptr )[0];
    // //load weights
    // #if LATTICE_HALF_PRECISION 
    //     __half* weights_ptr=reinterpret_cast<__half*>( &splatting_weights[level][idx][0]);
    //     __half4 weights_vec=reinterpret_cast<__half4*>( weights_ptr )[0];
    // #else 
    //     float* weights_ptr=reinterpret_cast<float*>( &splatting_weights[level][idx][0]);
    //     float4 weights_vec=reinterpret_cast<float4*>( weights_ptr )[0];
    // #endif

    // __half2 val_at_position;
    // val_at_position.x=grad_sliced_values_monolithic[level][0][idx];
    // val_at_position.y=grad_sliced_values_monolithic[level][1][idx];


 
    // //we assume 4 vertices and accumulate to them
    // // atomicAdd(&lattice_values_monolithic[level][0][indices_vec.x],   val_at_position.x* weights_vec.x  );
    // atomicAdd(static_cast<__half2*>((void*)&lattice_values_monolithic[level][indices_vec.x][0]),   val_at_position.x* weights_vec.x  );


}

template<int pos_dim, int val_dim>
__global__ void 
__launch_bounds__(BLOCK_SIZE_BACK) //since the block size is known at compile time we can specify it to the kernel and therefore cuda doesnt need to use heuristics based on code complexity to minimize registry usage
slice_backwards_no_precomputation_no_homogeneous_mr_monolithic(
    const int nr_positions,
    const int lattice_capacity,
    const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> lattice_values_monolithic,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> positions,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> scale_factor,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> random_shift_monolithic,
    const torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> anneal_window,
    // const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> grad_sliced_values_monolithic,
    // #if LATTICE_HALF_PRECISION
        // const torch::PackedTensorAccessor32<at::Half,3,torch::RestrictPtrTraits> grad_sliced_values_monolithic,
    // #else
        const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> grad_sliced_values_monolithic,
    // #endif
    // torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> lattice_values_monolithic
    #if LATTICE_HALF_PRECISION 
        torch::PackedTensorAccessor32<at::Half,3,torch::RestrictPtrTraits> lattice_values_monolithic_grad,
    #else 
        torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> lattice_values_monolithic_grad,
    #endif
    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> positions_grad,
    const bool concat_points,
    const bool require_lattice_values_grad,
    const bool require_positions_grad
    ){


    //values_vertices refers to the values that the lattice had in the forward pass. it has size m_hash_table_capcity x (val_dim+1)
    //grad_sliced_values is the gradient of the loss with respect to the sliced out values which has size nr_positions x val_dim
    const int idx = blockIdx.x * blockDim.x + threadIdx.x; // each thread will deal with one position
    if(idx >= nr_positions){
        return;
    }

    const uint32_t level = blockIdx.y; // <- the level is the same for all threads



    //get the value at the position
    float grad_sliced_val_cur[val_dim];
    #if LATTICE_HALF_PRECISION 
        #pragma unroll
        for (int j = 0; j < val_dim; j++) {
            grad_sliced_val_cur[j]=__half2float(grad_sliced_values_monolithic[level][j][idx]);
        }
    #else 
        #pragma unroll
        for (int j = 0; j < val_dim; j++) {
            grad_sliced_val_cur[j]=grad_sliced_values_monolithic[level][j][idx];
        }
    #endif







    float elevated[pos_dim + 1];


    float sm = 0;
    #pragma unroll
    for (int i = pos_dim; i > 0; i--) {
        float cf = (positions[idx][i-1] +random_shift_monolithic[level][i-1]  ) * scale_factor[level][i - 1];
        // float cf = positions[idx][i-1] * scalings_constants[(i - 1)  + level*3];
        // float cf = positions[idx][i-1] * scalings[(i - 1)  + level*3];
        // float cf = positions[i-1][idx] * scale_factor[level][i - 1];
        elevated[i] = sm - i * cf;
        sm += cf;
    }
    elevated[0] = sm;


    int rem0[pos_dim + 1];
    int rank[pos_dim + 1]{0};




    // Find the closest 0-colored simplex through rounding
    // greedily search for the closest zero-colored lattice point
    int sum = 0;
    #pragma unroll
    for (int i = 0; i <= pos_dim; i++) {
        float v = elevated[i] * (1.0 / (pos_dim + 1));
        float up = ceil(v) * (pos_dim + 1);
        float down = floor(v) * (pos_dim + 1);
        if (up - elevated[i] < elevated[i] - down) {
            rem0[i] = (int) up;
        } else {
            rem0[i] = (int) down;
        }
        sum += rem0[i];
    }
    sum /= pos_dim + 1;


    // Find the simplex we are in and store it in rank (where rank describes what position coordinate i has in the sorted order of the features values)
    #pragma unroll
    for (int i = 0; i < pos_dim; i++) {
        double di = elevated[i] - rem0[i];
        for (int j = i + 1; j <= pos_dim; j++)
            if (di < elevated[j] - rem0[j])
                rank[i]++;
            else
                rank[j]++;
    }

    // If the point doesn't lie on the plane (sum != 0) bring it back
    #pragma unroll
    for (int i = 0; i <= pos_dim; i++) {
        rank[i] += sum;
        if (rank[i] < 0) {
            rank[i] += pos_dim + 1;
            rem0[i] += pos_dim + 1;
        } else if (rank[i] > pos_dim) {
            rank[i] -= pos_dim + 1;
            rem0[i] -= pos_dim + 1;
        }
    }



    float barycentric[pos_dim + 2]{0};
    // Compute the barycentric coordinates (p.10 in [Adams etal 2010])
    #pragma unroll
    for (int i = 0; i <= pos_dim; i++) {
        float delta = (elevated[i] - rem0[i]) * (1.0 / (pos_dim + 1));
        barycentric[pos_dim - rank[i]] += delta;
        barycentric[pos_dim + 1 - rank[i]] -= delta;
    }
    // Wrap around
    barycentric[0] += 1.0 + barycentric[pos_dim + 1];



    //attempt 2 like in page 86/146 https://core.ac.uk/download/pdf/85209106.pdf
    // for (int i = 0; i <= pos_dim; i++) {
    //     barycentric[i]=barycentric_to_c1_continous(barycentric[i]);
    // }
    // //renormalize
    // float sum_bar=0.0;
    // for (int i = 0; i <= pos_dim; i++) {
    //     sum_bar+=barycentric[i];
    // }
    // // printf("sum %f \n", sum_bar);
    // for (int i = 0; i <= pos_dim; i++) {
    //     barycentric[i]/=sum_bar;
    // }




    //here we accumulate the values and the homogeneous term
    // float val_hom[val_dim]{0};

    float w_lvl= anneal_window[level];


    //get the value at the position
    // float grad_pos[val_dim];
    // #pragma unroll
    // for (int j = 0; j < val_dim; j++) {
    //     grad_pos[j]=grad_sliced_values_monolithic[level][j][idx];
    // }
    


    int key[pos_dim];
    if (require_lattice_values_grad){
        #pragma unroll
        for (int remainder = 0; remainder <= pos_dim; remainder++) {
            // Compute the location of the lattice point explicitly (all but
            // the last coordinate - it's redundant because they sum to zero)
            #pragma unroll
            for (int i = 0; i < pos_dim; i++) {
                key[i] = rem0[i] + remainder;
                if (rank[i] > pos_dim - remainder)
                    key[i] -= (pos_dim + 1);
            }

            // Retrieve pointer to the value at this vertex.
            int idx_val=idx_hash_with_collision<pos_dim>(key, lattice_capacity);


            float w= barycentric[remainder] * w_lvl; 

            //do it with only one half2 accumulate 
            #if LATTICE_HALF_PRECISION 
                __half2 weighted_grad;
                weighted_grad.x=__float2half(grad_sliced_val_cur[0]* w);
                weighted_grad.y=__float2half(grad_sliced_val_cur[1]* w);
                __half2* ptr= static_cast<__half2*>((void*)&lattice_values_monolithic_grad[level][idx_val][0]);
                atomicAdd(ptr, weighted_grad  ); 
            #else 


                //if the vertex exists accumulate its value weighted by the barycentric weight (accumulates also the homogeneous coordinate)
                #pragma unroll
                for (int j = 0; j < val_dim ; j++){
                    // val_hom[i] += lattice_values_monolithic[level][idx_val][i] * barycentric[remainder];
                    // val_hom[i] += lattice_values_monolithic[level][i][idx_val] * barycentric[remainder];
                    float weighted_grad=grad_sliced_val_cur[j]*w;
                    atomicAdd(&lattice_values_monolithic_grad[level][idx_val][j], weighted_grad  );
                    // lattice_values_monolithic[level][j][idx_val]=weighted_grad;


                    //try to make the atomic add faster like : https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/cuda/GridSampler.cuh
                    //slice_back_cuda_with train_sdf and 24 levels takes 3.6 ms
                    //DOES NOT make sense to use this because this is only a special thing for values of types half. For things fo type float still just an atomic add is the best 
                    // unsigned int memory_span = 24*lattice_capacity*2; // it is just the number of elements as shown in https://github.com/pytorch/pytorch/blob/91a5f52f51de9d6aa305d184fe07fe15d20b82c9/aten/src/ATen/native/cuda/GridSampler.cu
                    // fastSpecializedAtomicAdd(lattice_values_monolithic.data(),
                    //   level*lattice_capacity*val_dim + idx_val*val_dim+j,
                    //   memory_span,
                    //   weighted_grad);

                }

            #endif
        
        }
    }


    // bool debug=false;
    // if (idx==0){
        // debug=true;
    // }



    if(require_positions_grad){
        //We have from upstrema grad the dL/dS which is the derivative of the loss wrt to the sliced value
        //If we require positions grad we want to obtain dL/dPos
        //dL/dPos = dL/dS *dS/dB * dB/dE * dE/dPos
        //We need dS/dB which is the derivative of the sliced value wrt to the barycentric coords
        //We need dB/dE which is the derivative of the barycentric wrt to the elevated value
        //We need dE/dP which is the derivative of the elevated wrt to the position in xyz

        //dL/dB  = dL/dS *dS/dB 
        //foward pass is just S=B0*WLvl*V0 + B1*WLvl*V1 etc
        //so dS/dB0 is just W*V0
        float dL_dbarycentric[pos_dim + 2]{0};
        for (int remainder = 0; remainder <= pos_dim; remainder++) {
            //TODO maybe this can be sped up by doing it in the same loop as the lattice values gradient
            // Compute the location of the lattice point explicitly (all but
            // the last coordinate - it's redundant because they sum to zero)
            #pragma unroll
            for (int i = 0; i < pos_dim; i++) {
                key[i] = rem0[i] + remainder;
                if (rank[i] > pos_dim - remainder)
                    key[i] -= (pos_dim + 1);
            }
            // Retrieve pointer to the value at this vertex.
            int idx_val=idx_hash_with_collision<pos_dim>(key, lattice_capacity);

            //Load the value for this vertex
            const float* fv=&lattice_values_monolithic[level][idx_val][0];
            const float2 val_lattice_vertex=reinterpret_cast<const float2*>( fv )[0];
            //add to the dL_d_barycentric
            dL_dbarycentric[remainder]+=val_lattice_vertex.x*w_lvl   * grad_sliced_val_cur[0];
            dL_dbarycentric[remainder]+=val_lattice_vertex.y*w_lvl   * grad_sliced_val_cur[1];

        }
        // if(debug) printf("grad sliced is %f, %f\n", grad_sliced_val_cur[0], grad_sliced_val_cur[1]);
        // if(debug) printf("dL_dbarycentric[0] %f, dL_dbarycentric[1] %f, dL_dbarycentric[2] %f, dL_dbarycentric[3] %f\n", dL_dbarycentric[0], dL_dbarycentric[1], dL_dbarycentric[2], dL_dbarycentric[3]);

        //dL/dE  = dL/dB *dB/dE
        //In the forward pass of computing B from E there is this wraparound line of barycentric[0] += 1.0 + barycentric[pos_dim + 1];
        // barycentric[0] = barycentric[0]+ 1.0 + barycentric[pos_dim + 1];
        //I think this means that the gradient of also added to barycentric{pos_dim+1}
        //TODO check for correctness here
        dL_dbarycentric[pos_dim + 1] += dL_dbarycentric[0]; //order here is important btw, we first add B0 to B5 and only afterwards we double B0
        // dL_dbarycentric[0]=dL_dbarycentric[0]*2;
        //Now we need to accumulate gradient into elevated from from each barycentric that the particlar elevated affected
        float dL_delevated[pos_dim + 1]{0};
        #pragma unroll
        for (int i = 0; i <= pos_dim; i++) {
            dL_delevated[i]+=  dL_dbarycentric[pos_dim - rank[i]] * (1.0 / (pos_dim + 1));
            dL_delevated[i]-=  dL_dbarycentric[pos_dim + 1 - rank[i]] * (1.0 / (pos_dim + 1));
        }
        // if(debug) printf("dL_delevated[0] %f, dL_delevated[1] %f, dL_delevated[2] %f, dL_delevated[3] %f\n", dL_delevated[0], dL_delevated[1], dL_delevated[2], dL_delevated[3]);

        //dL/dPos = dL/dE * dE/dPos
        float dL_dPos[pos_dim];
        //I unrolles the loop that computes E from P and I got some local derivatives like 
        //dEx/dPx=Sx  dEx/dPy=Sy
        //dEy/dPx=-Sx  dEy/dPy=Sy  dEy/dPz=Sz
        //dEz/dPy=-2Sy  dEz/dPz=Sz
        //dEw/dPz=-3Sz
        //So we just accumulate these values inot dL_dPos
        //x
        dL_dPos[0]= dL_delevated[0]* scale_factor[level][0] +  
                    dL_delevated[1]* (-scale_factor[level][0]);
        //y
        dL_dPos[1]= dL_delevated[0]* scale_factor[level][1] +  
                    dL_delevated[1]* scale_factor[level][1] +
                    dL_delevated[2]* (-2*scale_factor[level][1]);
        //z
        dL_dPos[2]= dL_delevated[0]* scale_factor[level][2] + 
                    dL_delevated[1]* scale_factor[level][2] +
                    dL_delevated[2]* scale_factor[level][2] +
                    dL_delevated[3]* (-3*scale_factor[level][2]);
        // if(debug) printf("dL_dPos[0] %f, dL_dPos[1] %f, dL_dPos[2] %f\n", dL_dPos[0], dL_dPos[1], dL_dPos[2]);
        //finish
        // printf("dL_dPos[0] %f \n",dL_dPos[0]);
        atomicAdd(&positions_grad[idx][0], dL_dPos[0]  );
        atomicAdd(&positions_grad[idx][1], dL_dPos[1]  );
        atomicAdd(&positions_grad[idx][2], dL_dPos[2]  );
        //Cannot be done like this because the sums into the positions grad may come from multiple levels so they need to be atomic
        // positions_grad[idx][0]=dL_dPos[0];
        // positions_grad[idx][1]=dL_dPos[1];
        // positions_grad[idx][2]=dL_dPos[2];
                    

    }

   
   


}


//double back
template<int pos_dim, int val_dim>
__global__ void 
__launch_bounds__(BLOCK_SIZE_BACK) //since the block size is known at compile time we can specify it to the kernel and therefore cuda doesnt need to use heuristics based on code complexity to minimize registry usage
slice_double_back_from_positions_grad_gpu(
    const int nr_positions,
    const int lattice_capacity,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> double_positions_grad,
    const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> lattice_values_monolithic,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> positions,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> scale_factor,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> random_shift_monolithic,
    const torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> anneal_window,
    const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> grad_sliced_values_monolithic,
    const bool concat_points,
    //output
    torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> grad_grad_sliced_values_monolithic,
    #if LATTICE_HALF_PRECISION 
        torch::PackedTensorAccessor32<at::Half,3,torch::RestrictPtrTraits> lattice_values_monolithic_grad
    #else 
        torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> lattice_values_monolithic_grad
    #endif
    ){


    //values_vertices refers to the values that the lattice had in the forward pass. it has size m_hash_table_capcity x (val_dim+1)
    //grad_sliced_values is the gradient of the loss with respect to the sliced out values which has size nr_positions x val_dim
    const int idx = blockIdx.x * blockDim.x + threadIdx.x; // each thread will deal with one position
    if(idx >= nr_positions){
        return;
    }

    const uint32_t level = blockIdx.y; // <- the level is the same for all threads











    float elevated[pos_dim + 1];


    float sm = 0;
    #pragma unroll
    for (int i = pos_dim; i > 0; i--) {
        float cf = (positions[idx][i-1] +random_shift_monolithic[level][i-1]  ) * scale_factor[level][i - 1];
        // float cf = positions[idx][i-1] * scalings_constants[(i - 1)  + level*3];
        // float cf = positions[idx][i-1] * scalings[(i - 1)  + level*3];
        // float cf = positions[i-1][idx] * scale_factor[level][i - 1];
        elevated[i] = sm - i * cf;
        sm += cf;
    }
    elevated[0] = sm;


    int rem0[pos_dim + 1];
    int rank[pos_dim + 1]{0};




    // Find the closest 0-colored simplex through rounding
    // greedily search for the closest zero-colored lattice point
    int sum = 0;
    #pragma unroll
    for (int i = 0; i <= pos_dim; i++) {
        float v = elevated[i] * (1.0 / (pos_dim + 1));
        float up = ceil(v) * (pos_dim + 1);
        float down = floor(v) * (pos_dim + 1);
        if (up - elevated[i] < elevated[i] - down) {
            rem0[i] = (int) up;
        } else {
            rem0[i] = (int) down;
        }
        sum += rem0[i];
    }
    sum /= pos_dim + 1;


    // Find the simplex we are in and store it in rank (where rank describes what position coordinate i has in the sorted order of the features values)
    #pragma unroll
    for (int i = 0; i < pos_dim; i++) {
        double di = elevated[i] - rem0[i];
        for (int j = i + 1; j <= pos_dim; j++)
            if (di < elevated[j] - rem0[j])
                rank[i]++;
            else
                rank[j]++;
    }

    // If the point doesn't lie on the plane (sum != 0) bring it back
    #pragma unroll
    for (int i = 0; i <= pos_dim; i++) {
        rank[i] += sum;
        if (rank[i] < 0) {
            rank[i] += pos_dim + 1;
            rem0[i] += pos_dim + 1;
        } else if (rank[i] > pos_dim) {
            rank[i] -= pos_dim + 1;
            rem0[i] -= pos_dim + 1;
        }
    }



    float barycentric[pos_dim + 2]{0};
    // Compute the barycentric coordinates (p.10 in [Adams etal 2010])
    #pragma unroll
    for (int i = 0; i <= pos_dim; i++) {
        float delta = (elevated[i] - rem0[i]) * (1.0 / (pos_dim + 1));
        barycentric[pos_dim - rank[i]] += delta;
        barycentric[pos_dim + 1 - rank[i]] -= delta;
    }
    // Wrap around
    barycentric[0] += 1.0 + barycentric[pos_dim + 1];



    

    float w_lvl= anneal_window[level];

    //get the value at the position
    float grad_sliced_val_cur[val_dim];
    #if LATTICE_HALF_PRECISION 
        #pragma unroll
        for (int j = 0; j < val_dim; j++) {
            grad_sliced_val_cur[j]=__half2float(grad_sliced_values_monolithic[level][j][idx]);
        }
    #else 
        #pragma unroll
        for (int j = 0; j < val_dim; j++) {
            grad_sliced_val_cur[j]=grad_sliced_values_monolithic[level][j][idx];
        }
    #endif

    //get eh gradient at the curent position
    float grad_p_cur[pos_dim];
    #pragma unroll
    for (int j = 0; j < pos_dim; j++) {
        grad_p_cur[j]=double_positions_grad[idx][j];
    }



    int key[pos_dim];


    //We have upstream gradient dL/dPos which is double_positions_grad
    //we want dL/dV and dL/dS, so we want to push the gradient into lattice_values_monolithic_grad    grad_grad_sliced_values_monolithic
    // dL/dS = dL/dP * dP/dE * dE/dB * dB/dS
    // dL/dV = dL/dP * dP/dE * dE/dB * dB/dV
    //STARTING
    // dP/dE 
    float dL_delevated[pos_dim + 1]{0};
    dL_delevated[0] =   grad_p_cur[0] * scale_factor[level][0] + 
                        grad_p_cur[1] * scale_factor[level][1] +
                        grad_p_cur[2] * scale_factor[level][2];
    dL_delevated[1] =   grad_p_cur[0] * (-scale_factor[level][0]) + 
                        grad_p_cur[1] * scale_factor[level][1] +
                        grad_p_cur[2] * scale_factor[level][2];
    dL_delevated[2] =   grad_p_cur[1] * (-2*scale_factor[level][1]) +
                        grad_p_cur[2] * scale_factor[level][2];
    dL_delevated[3] =   grad_p_cur[2] * (-3*scale_factor[level][2]);
    // dE/dB
    float dL_dbarycentric[pos_dim + 2]{0};
    //in the forward pass we did:
    // dL_dbarycentric[pos_dim + 1] += dL_dbarycentric[0]; //order here is important btw, we first add B0 to B5 and only afterwards we double B0
    // dL_dbarycentric[0]=dL_dbarycentric[0]*2;
    // float dL_delevated[pos_dim + 1]{0};
    // #pragma unroll
    // for (int i = 0; i <= pos_dim; i++) {
    //     dL_delevated[i]+=  dL_dbarycentric[pos_dim - rank[i]] * (1.0 / (pos_dim + 1));
    //     dL_delevated[i]-=  dL_dbarycentric[pos_dim + 1 - rank[i]] * (1.0 / (pos_dim + 1));
    // }
    //So now we do this
    for (int i = 0; i <= pos_dim; i++) {
        dL_dbarycentric[pos_dim - rank[i]] += dL_delevated[i]* (1.0 / (pos_dim + 1));
        dL_dbarycentric[pos_dim + 1 - rank[i]] -= dL_delevated[i]* (1.0 / (pos_dim + 1));
    }
    // dL_dbarycentric[0]=dL_dbarycentric[0]*2;
    dL_dbarycentric[0] += dL_dbarycentric[pos_dim + 1];
    //push gradient into values_lattice and grad_sliced
    float grad_grad_sliced_val_cur[val_dim]{0};
    for (int remainder = 0; remainder <= pos_dim; remainder++) {
        //TODO maybe this can be sped up by doing it in the same loop as the lattice values gradient
        // Compute the location of the lattice point explicitly (all but
        // the last coordinate - it's redundant because they sum to zero)
        #pragma unroll
        for (int i = 0; i < pos_dim; i++) {
            key[i] = rem0[i] + remainder;
            if (rank[i] > pos_dim - remainder)
                key[i] -= (pos_dim + 1);
        }
        // Retrieve pointer to the value at this vertex.
        int idx_val=idx_hash_with_collision<pos_dim>(key, lattice_capacity);

        //Load the value for this vertex
        const float* fv=&lattice_values_monolithic[level][idx_val][0];
        const float2 val_lattice_vertex=reinterpret_cast<const float2*>( fv )[0];
        //add to the dL_d_barycentric
        // dL_dbarycentric[remainder]+=val_lattice_vertex.x*w_lvl   * grad_sliced_val_cur[0];
        // dL_dbarycentric[remainder]+=val_lattice_vertex.y*w_lvl   * grad_sliced_val_cur[1];
        // lattice_values_monolithic_grad[level][idx_val][0] += dL_dbarycentric[remainder]* w_lvl * grad_sliced_val_cur[0];
        // lattice_values_monolithic_grad[level][idx_val][1] += dL_dbarycentric[remainder]* w_lvl * grad_sliced_val_cur[1];
        atomicAdd(&lattice_values_monolithic_grad[level][idx_val][0], dL_dbarycentric[remainder]* w_lvl * grad_sliced_val_cur[0]  );
        atomicAdd(&lattice_values_monolithic_grad[level][idx_val][1], dL_dbarycentric[remainder]* w_lvl * grad_sliced_val_cur[1]  );


        // push gradient into grad_sliced_val_cur
        // grad_sliced_val_cur add towards all the barycentric coord, so in the backward pass the gradient from b0 to all the grad_sliced_val_cur
        grad_grad_sliced_val_cur[0]+=dL_dbarycentric[remainder]* w_lvl * val_lattice_vertex.x;
        grad_grad_sliced_val_cur[1]+=dL_dbarycentric[remainder]* w_lvl * val_lattice_vertex.y;
    }
    //finish the accumulation of grad_grad_sliced
    atomicAdd(&grad_grad_sliced_values_monolithic[level][0][idx], grad_grad_sliced_val_cur[0]  );
    atomicAdd(&grad_grad_sliced_values_monolithic[level][1][idx], grad_grad_sliced_val_cur[1]  );




    // if (require_lattice_values_grad){
    //     #pragma unroll
    //     for (int remainder = 0; remainder <= pos_dim; remainder++) {
    //         // Compute the location of the lattice point explicitly (all but
    //         // the last coordinate - it's redundant because they sum to zero)
    //         #pragma unroll
    //         for (int i = 0; i < pos_dim; i++) {
    //             key[i] = rem0[i] + remainder;
    //             if (rank[i] > pos_dim - remainder)
    //                 key[i] -= (pos_dim + 1);
    //         }

    //         // Retrieve pointer to the value at this vertex.
    //         int idx_val=idx_hash_with_collision<pos_dim>(key, lattice_capacity);


    //         float w= barycentric[remainder] * w_lvl; 

    //         //do it with only one half2 accumulate 
    //         #if LATTICE_HALF_PRECISION 
    //             __half2 weighted_grad;
    //             weighted_grad.x=__float2half(grad_sliced_val_cur[0]* w);
    //             weighted_grad.y=__float2half(grad_sliced_val_cur[1]* w);
    //             __half2* ptr= static_cast<__half2*>((void*)&lattice_values_monolithic_grad[level][idx_val][0]);
    //             atomicAdd(ptr, weighted_grad  ); 
    //         #else 


    //             //if the vertex exists accumulate its value weighted by the barycentric weight (accumulates also the homogeneous coordinate)
    //             #pragma unroll
    //             for (int j = 0; j < val_dim ; j++){
    //                 // val_hom[i] += lattice_values_monolithic[level][idx_val][i] * barycentric[remainder];
    //                 // val_hom[i] += lattice_values_monolithic[level][i][idx_val] * barycentric[remainder];
    //                 float weighted_grad=grad_sliced_val_cur[j]*w;
    //                 atomicAdd(&lattice_values_monolithic_grad[level][idx_val][j], weighted_grad  );
    //                 // lattice_values_monolithic[level][j][idx_val]=weighted_grad;


    //                 //try to make the atomic add faster like : https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/cuda/GridSampler.cuh
    //                 //slice_back_cuda_with train_sdf and 24 levels takes 3.6 ms
    //                 //DOES NOT make sense to use this because this is only a special thing for values of types half. For things fo type float still just an atomic add is the best 
    //                 // unsigned int memory_span = 24*lattice_capacity*2; // it is just the number of elements as shown in https://github.com/pytorch/pytorch/blob/91a5f52f51de9d6aa305d184fe07fe15d20b82c9/aten/src/ATen/native/cuda/GridSampler.cu
    //                 // fastSpecializedAtomicAdd(lattice_values_monolithic.data(),
    //                 //   level*lattice_capacity*val_dim + idx_val*val_dim+j,
    //                 //   memory_span,
    //                 //   weighted_grad);

    //             }

    //         #endif
        
    //     }
    // }


    

    // if(require_positions_grad){
    //     //We have from upstrema grad the dL/dS which is the derivative of the loss wrt to the sliced value
    //     //If we require positions grad we want to obtain dL/dPos
    //     //dL/dPos = dL/dS *dS/dB * dB/dE * dE/dPos
    //     //We need dS/dB which is the derivative of the sliced value wrt to the barycentric coords
    //     //We need dB/dE which is the derivative of the barycentric wrt to the elevated value
    //     //We need dE/dP which is the derivative of the elevated wrt to the position in xyz

    //     //dL/dB  = dL/dS *dS/dB 
    //     //foward pass is just S=B0*WLvl*V0 + B1*WLvl*V1 etc
    //     //so dS/dB0 is just W*V0
    //     float dL_dbarycentric[pos_dim + 2]{0};
    //     for (int remainder = 0; remainder <= pos_dim; remainder++) {
    //         //TODO maybe this can be sped up by doing it in the same loop as the lattice values gradient
    //         // Compute the location of the lattice point explicitly (all but
    //         // the last coordinate - it's redundant because they sum to zero)
    //         #pragma unroll
    //         for (int i = 0; i < pos_dim; i++) {
    //             key[i] = rem0[i] + remainder;
    //             if (rank[i] > pos_dim - remainder)
    //                 key[i] -= (pos_dim + 1);
    //         }
    //         // Retrieve pointer to the value at this vertex.
    //         int idx_val=idx_hash_with_collision<pos_dim>(key, lattice_capacity);

    //         //Load the value for this vertex
    //         const float* fv=&lattice_values_monolithic[level][idx_val][0];
    //         const float2 val_lattice_vertex=reinterpret_cast<const float2*>( fv )[0];
    //         //add to the dL_d_barycentric
    //         dL_dbarycentric[remainder]+=val_lattice_vertex.x*w_lvl   * grad_sliced_val_cur[0];
    //         dL_dbarycentric[remainder]+=val_lattice_vertex.y*w_lvl   * grad_sliced_val_cur[1];

    //     }
    //     // if(debug) printf("grad sliced is %f, %f\n", grad_sliced_val_cur[0], grad_sliced_val_cur[1]);
    //     // if(debug) printf("dL_dbarycentric[0] %f, dL_dbarycentric[1] %f, dL_dbarycentric[2] %f, dL_dbarycentric[3] %f\n", dL_dbarycentric[0], dL_dbarycentric[1], dL_dbarycentric[2], dL_dbarycentric[3]);

    //     //dL/dE  = dL/dB *dB/dE
    //     //In the forward pass of computing B from E there is this wraparound line of barycentric[0] += 1.0 + barycentric[pos_dim + 1];
    //     //I think this means that the gradient of B[0] gets duplicates in B[0] and also added to barycentric{pos_dim+1}
    //     //TODO check for correctness here
    //     dL_dbarycentric[pos_dim + 1] += dL_dbarycentric[0]; //order here is important btw, we first add B0 to B5 and only afterwards we double B0
    //     dL_dbarycentric[0]=dL_dbarycentric[0]*2;
    //     //Now we need to accumulate gradient into elevated from from each barycentric that the particlar elevated affected
    //     float dL_delevated[pos_dim + 1]{0};
    //     #pragma unroll
    //     for (int i = 0; i <= pos_dim; i++) {
    //         dL_delevated[i]+=  dL_dbarycentric[pos_dim - rank[i]] * (1.0 / (pos_dim + 1));
    //         dL_delevated[i]-=  dL_dbarycentric[pos_dim + 1 - rank[i]] * (1.0 / (pos_dim + 1));
    //     }
    //     // if(debug) printf("dL_delevated[0] %f, dL_delevated[1] %f, dL_delevated[2] %f, dL_delevated[3] %f\n", dL_delevated[0], dL_delevated[1], dL_delevated[2], dL_delevated[3]);

    //     //dL/dPos = dL/dE * dE/dPos
    //     float dL_dPos[pos_dim];
    //     //I unrolles the loop that computes E from P and I got some local derivatives like 
    //     //dEx/dPx=Sx  dEx/dPy=Sy
    //     //dEy/dPx=-Sx  dEy/dPy=Sy  dEy/dPz=Sz
    //     //dEz/dPy=-2Sy  dEz/dPz=Sz
    //     //dEw/dPz=-3Sz
    //     //So we just accumulate these values inot dL_dPos
    //     //x
    //     dL_dPos[0]= dL_delevated[0]* scale_factor[level][0] +  
    //                 dL_delevated[1]* (-scale_factor[level][0]);
    //     //y
    //     dL_dPos[1]= dL_delevated[0]* scale_factor[level][1] +  
    //                 dL_delevated[1]* scale_factor[level][1] +
    //                 dL_delevated[2]* (-2*scale_factor[level][1]);
    //     //z
    //     dL_dPos[2]= dL_delevated[0]* scale_factor[level][2] + 
    //                 dL_delevated[1]* scale_factor[level][2] +
    //                 dL_delevated[2]* scale_factor[level][2] +
    //                 dL_delevated[3]* (-3*scale_factor[level][2]);
    //     // if(debug) printf("dL_dPos[0] %f, dL_dPos[1] %f, dL_dPos[2] %f\n", dL_dPos[0], dL_dPos[1], dL_dPos[2]);
    //     //finish
    //     // printf("dL_dPos[0] %f \n",dL_dPos[0]);
    //     atomicAdd(&positions_grad[idx][0], dL_dPos[0]  );
    //     atomicAdd(&positions_grad[idx][1], dL_dPos[1]  );
    //     atomicAdd(&positions_grad[idx][2], dL_dPos[2]  );
    //     //Cannot be done like this because the sums into the positions grad may come from multiple levels so they need to be atomic
    //     // positions_grad[idx][0]=dL_dPos[0];
    //     // positions_grad[idx][1]=dL_dPos[1];
    //     // positions_grad[idx][2]=dL_dPos[2];
                    

    // }

   
   


}


//similar to https://github.com/MiguelMonteiro/permutohedral_lattice/blob/master/src/PermutohedralLatticeGPU.cuh
template<int pos_dim, int val_dim>
__global__ void 
__launch_bounds__(BLOCK_SIZE) //since the block size is known at compile time we can specify it to the kernel and therefore cuda doesnt need to use heuristics based on code complexity to minimize registry usage
slice_backwards_with_precomputation_no_homogeneous_mr_monolithic_sharedmem(
    const int nr_positions,
    const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> grad_sliced_values_monolithic,
    const torch::PackedTensorAccessor32<int,3,torch::RestrictPtrTraits> splatting_indices,
    const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> splatting_weights,
    torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> lattice_values_monolithic
    ){


    //values_vertices refers to the values that the lattice had in the forward pass. it has size m_hash_table_capcity x (val_dim+1)
    //grad_sliced_values is the gradient of the loss with respect to the sliced out values which has size nr_positions x val_dim
    const int idx = blockIdx.x * blockDim.x + threadIdx.x; // each thread will deal with one position
    const int threadId = threadIdx.x;
    const int color = blockIdx.y;
    const uint32_t level = blockIdx.z; // <- the level is the same for all threads
    const bool outOfBounds = (idx >= nr_positions);




    //attempt 4
    __shared__ int sh_splatting_indices[BLOCK_SIZE]; //for every position record the splatting idx
    __shared__ float sh_weighted_vals[BLOCK_SIZE * val_dim];
    int splatting_idx_cur_position = -1;
    float *sh_value_cur_position = sh_weighted_vals + threadId * val_dim;

    if (!outOfBounds) {
        int splatting_idx = splatting_indices[level][color][idx];
        float splatting_weight = splatting_weights[level][color][idx];

       
        sh_splatting_indices[threadId] = splatting_idx;
        splatting_idx_cur_position=splatting_idx;

        for (int j = 0; j < val_dim; j++){
            sh_value_cur_position[j]= grad_sliced_values_monolithic[level][j][idx] * splatting_weight;

        }
    } else {
        sh_splatting_indices[threadId] = -1;
    }

    __syncthreads(); //now all the shared things are full. Some of the values actually belong to the same lattice vertex and should be summed together

    // am I the first thread in this block to care about this key?
    if (outOfBounds)
        return;


    for (int i = 0; i < BLOCK_SIZE; i++) {
        if (i < threadId) {
            if (splatting_idx_cur_position == sh_splatting_indices[i]) {
                // somebody else with higher priority cares about this key
                return;
            }
        } else if (i > threadId) {
            if (splatting_idx_cur_position == sh_splatting_indices[i]) {
                // someone else with lower priority cares about this key, accumulate it into mine
                for (int j = 0; j < val_dim; j++) {
                    sh_weighted_vals[threadId * val_dim + j] += sh_weighted_vals[i * val_dim + j];
                }
            }
        }
    }

    // only the threads with something to write to main memory are still going
    if ( splatting_idx_cur_position >0 ){
        for (int j = 0; j < val_dim; j++) {
            atomicAdd(&lattice_values_monolithic[level][j][splatting_idx_cur_position], sh_value_cur_position[j]  );
        }
    }







    //attempt3 with normal atomicadd but done with a block of dim 3 THIS WORKS
    // if (outOfBounds) {
        // return;
    // }
    // int splatting_idx = splatting_indices[level][color][idx];
    // if(splatting_idx>=0){

    //     float weight = splatting_weights[level][color][idx];

    //     //acumulate the values
    //     #pragma unroll
    //     for (int j = 0; j < val_dim; j++) {

    //         float weighted_grad=grad_sliced_values_monolithic[level][j][idx]*weight;
    //         if(fabs(weighted_grad)>1e-15){
    //             atomicAdd(&lattice_values_monolithic[level][j][splatting_idx], weighted_grad  );
    //         }
            
    //     }
    // }






}


//similar to but instead of each thread doing a certain lvl, each thread accumulates gradient over the full val_dim and all the lvls, this makes it easier to accumulate gradient into positions
template<int pos_dim, int val_dim>
__global__ void 
__launch_bounds__(BLOCK_SIZE) //since the block size is known at compile time we can specify it to the kernel and therefore cuda doesnt need to use heuristics based on code complexity to minimize registry usage
slice_backwards_with_precomputation_no_homogeneous_mr_monolithic_full_pos(
    const int nr_positions,
    const int nr_resolutions,
    const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> grad_sliced_values_monolithic,
    const torch::PackedTensorAccessor32<int,3,torch::RestrictPtrTraits> splatting_indices,
    const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> splatting_weights,
    torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> lattice_values_monolithic
    ){


    //values_vertices refers to the values that the lattice had in the forward pass. it has size m_hash_table_capcity x (val_dim+1)
    //grad_sliced_values is the gradient of the loss with respect to the sliced out values which has size nr_positions x val_dim
    const int idx = blockIdx.x * blockDim.x + threadIdx.x; // each thread will deal with one position
    if(idx >= nr_positions){
        return;
    }

    // const uint32_t level = blockIdx.y; // <- the level is the same for all threads
    for(int level=0; level<nr_resolutions; level++){

    
        #pragma unroll 
        for(int color=0; color<pos_dim+1; color++){
            // int index_into_m_entries=round(splatting_indices_and_weights[ idx * (pos_dim + 1)*2 + color*2 + 0]);
            // int splatting_idx = splatting_indices[ idx * (pos_dim + 1) + color];

            int splatting_idx = splatting_indices[level][color][idx];
            // int splatting_idx = splatting_indices[level][idx][color];

            if(splatting_idx>=0){

                float weight = splatting_weights[level][color][idx];
                // float weight = splatting_weights[level][idx][color];
                

                //acumulate the values
                #pragma unroll
                for (int j = 0; j < val_dim; j++) {

                    float weighted_grad=grad_sliced_values_monolithic[level][j][idx]*weight;

                    if(fabs(weighted_grad)>1e-15){
                        // atomicAdd(&lattice_values_monolithic[level][splatting_idx][j], weighted_grad  );
                        atomicAdd(&lattice_values_monolithic[level][j][splatting_idx], weighted_grad  );
                        // #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 200)
                        // #else 
                        //     #warning CUDA ARCH NEEDS TO BE AT LEAST 200 IN ORDER TO ENABLE ATOMIC OPERATIONS!
                        // #endif
                    }
                
                
                }

            
            }


        }
    }


}










template<int pos_dim>
__global__ void 
elevate_points_gpu(
    const int nr_positions,  
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> positions,
    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> elevated
    ){

    // determine where in the thread grid we are
    int idx = blockIdx.x * blockDim.x + threadIdx.x; //each thread will deal with a new value

    if(idx>=nr_positions){ //don't go out of bounds
        return;
    }
    
    // float* elevated_point = elevated + idx * (pos_dim + 1);
    // const float *position = positions + idx * pos_dim;
    // elevate<pos_dim>(elevated_point, position);


    // float elevated[pos_dim + 1];
    // const float *position = positions + idx * pos_dim;
    // const float *position = positions + idx;
    // elevate<pos_dim>(elevated, position);

    //TODO the scale factor can be precomputed
    float scaleFactor[pos_dim];
    // float invStdDev = (pos_dim + 1) * sqrt(2.0f / 3);
    float invStdDev = 1.0;
    // float invStdDev = 1.0;
    for (int i = 0; i < pos_dim; i++) {
        scaleFactor[i] = 1.0f / (sqrt((float) (i + 1) * (i + 2))) * invStdDev;
    }

    // embed position vector into the hyperplane
    // first rotate position into the (pd+1)-dimensional hyperplane
    // sm contains the sum of 1..n of our feature vector
    float sm = 0;
    for (int i = pos_dim; i > 0; i--) {
        // float cf = position[i*nr_positions - 1*nr_positions] * scaleFactor[i - 1];
        float cf = positions[idx][i-1] * scaleFactor[i - 1];
        // float cf = position[i - 1] ;
        elevated[idx][i] = sm - i * cf;
        sm += cf;
    }
    elevated[idx][0] = sm;

}


template<int pos_dim>
__global__ void 
// __launch_bounds__(BLOCK_SIZE) //since the block size is known at compile time we can specify it to the kernel and therefore cuda doesnt need to use heuristics based on code complexity to minimize registry usage
create_non_differentiable_indices_for_slice_with_collisions_gpu(
    const int nr_positions,
    const int lattice_capacity,
    // const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> positions,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> positions_elevated,
    torch::PackedTensorAccessor32<int,2,torch::RestrictPtrTraits> rem0_matrix,
    torch::PackedTensorAccessor32<int,2,torch::RestrictPtrTraits> rank_matrix,
    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> splatting_indices
    // HashTableGPU hash_table
    ) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x; //each thread will deal with a new value


    if(idx>=nr_positions){ //don't go out of bounds
        return;
    }


    
    float elevated[pos_dim + 1];


    //calculate_elevated
    // float scaleFactor[pos_dim];
    // // float invStdDev = (pos_dim + 1) * sqrt(2.0f / 3);
    // float invStdDev = 1.0;
    // for (int i = 0; i < pos_dim; i++) {
    //     scaleFactor[i] = 1.0f / (sqrt((float) (i + 1) * (i + 2))) * invStdDev;
    // }
    // // embed position vector into the hyperplane
    // // first rotate position into the (pd+1)-dimensional hyperplane
    // // sm contains the sum of 1..n of our feature vector
    // float sm = 0;
    // for (int i = pos_dim; i > 0; i--) {
    //     // float cf = position[i*nr_positions - 1*nr_positions] * scaleFactor[i - 1];
    //     float cf = positions[idx][i-1] * scaleFactor[i - 1];
    //     // float cf = position[i - 1] ;
    //     elevated[i] = sm - i * cf;
    //     sm += cf;
    // }
    // elevated[0] = sm;


    for (int i = 0; i <= pos_dim; i++) { 
        elevated[i] = positions_elevated[idx][i];
    }






    int rem0[pos_dim + 1];
    int rank[pos_dim + 1]{0};


   


    // Find the closest 0-colored simplex through rounding
    // greedily search for the closest zero-colored lattice point
    int sum = 0;
    for (int i = 0; i <= pos_dim; i++) {
        float v = elevated[i] * (1.0 / (pos_dim + 1));
        float up = ceil(v) * (pos_dim + 1);
        float down = floor(v) * (pos_dim + 1);
        if (up - elevated[i] < elevated[i] - down) {
            rem0[i] = (int) up;
        } else {
            rem0[i] = (int) down;
        }
        sum += rem0[i];
    }
    sum /= pos_dim + 1;


    // Find the simplex we are in and store it in rank (where rank describes what position coordinate i has in the sorted order of the features values)
    for (int i = 0; i < pos_dim; i++) {
        double di = elevated[i] - rem0[i];
        for (int j = i + 1; j <= pos_dim; j++)
            if (di < elevated[j] - rem0[j])
                rank[i]++;
            else
                rank[j]++;
    }

    // If the point doesn't lie on the plane (sum != 0) bring it back
    for (int i = 0; i <= pos_dim; i++) {
        rank[i] += sum;
        if (rank[i] < 0) {
            rank[i] += pos_dim + 1;
            rem0[i] += pos_dim + 1;
        } else if (rank[i] > pos_dim) {
            rank[i] -= pos_dim + 1;
            rem0[i] -= pos_dim + 1;
        }
    }



    // float barycentric[pos_dim + 2]{0};
    // // Compute the barycentric coordinates (p.10 in [Adams etal 2010])
    // for (int i = 0; i <= pos_dim; i++) {
    //     float delta = (elevated[i] - rem0[i]) * (1.0 / (pos_dim + 1));
    //     barycentric[pos_dim - rank[i]] += delta;
    //     barycentric[pos_dim + 1 - rank[i]] -= delta;
    // }
    // // Wrap around
    // barycentric[0] += 1.0 + barycentric[pos_dim + 1];



    //here we accumulate the values and the homogeneous term
    // float val_hom[val_dim]{0};

    int key[pos_dim];
    for (int remainder = 0; remainder <= pos_dim; remainder++) {
        // Compute the location of the lattice point explicitly (all but
        // the last coordinate - it's redundant because they sum to zero)
        for (int i = 0; i < pos_dim; i++) {
            key[i] = static_cast<int>(rem0[i] + remainder);
            if (rank[i] > pos_dim - remainder)
                key[i] -= (pos_dim + 1);
        }

        // int idx_val=hash_table.retrieve_with_collisions(key);
        int idx_val=idx_hash_with_collision<pos_dim>(key, lattice_capacity);

        
        //store also the splatting indices and weight so that they can be used for the backwards pass
        splatting_indices[idx * (pos_dim + 1) + remainder]=idx_val; //it indexes in m_keys
       
    
    }

    //store also the rem0 and rank
    for (int i = 0; i <= pos_dim; i++) {
        rem0_matrix[idx][i]=rem0[i];
    }
    for (int i = 0; i <= pos_dim; i++) {
        rank_matrix[idx][i]=rank[i];
    }


   

   



}






