#pragma once


#include <torch/torch.h>



#define BLOCK_SIZE 128
#define BLOCK_SIZE_BACK 64
#define BLOCK_SIZE_DOUBLE_BACK 64

// #define LATTICE_HALF_PRECISION 0


__constant__ float random_shift_constant[256];
__constant__ float scale_factor_constant[256];




template<int pos_dim>
/* Hash function used in this implementation. A simple base conversion. */  
__forceinline__ __device__ unsigned int hash(const int *const key) {
    unsigned int k = 0;
    #pragma unroll
    for (int i = 0; i < pos_dim; i++) {
        k += key[i];
        k = k * 2531011;
    }
    return k;
}

__forceinline__ __device__ int modHash(const unsigned int& n, const int& capacity){
    return(n % capacity);
}

template<int pos_dim>
__forceinline__ __device__ int idx_hash_with_collision(const int * const key, const int& capacity) {

    int h = modHash(hash<pos_dim>(key), capacity);
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
__launch_bounds__(BLOCK_SIZE) //since the block size is known at compile time we can specify it to the kernel and therefore cuda doesnt need to use heuristics based on code complexity to minimize registry usage
forward_gpu(
    const int nr_positions,
    const int lattice_capacity,
    const int nr_resolutions,
    const int nr_resolutions_extra,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> positions,
    torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> lattice_values_monolithic,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> scale_factor,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> random_shift_monolithic,
    const torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> anneal_window,
    torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> sliced_values_monolithic,
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
                sliced_values_monolithic[level][i][idx]=0.0f;
                // printf("putting 0 at dimension %d\n", i);
            }
        }



        return;

    }


    
    float elevated[pos_dim + 1];

    


    float sm = 0;
    #pragma unroll
    for (int i = pos_dim; i > 0; i--) {
        // float cf = (positions[idx][i-1] +random_shift_monolithic[level][i-1]  ) * scale_factor[level][i - 1];
        float cf = (positions[idx][i-1] +random_shift_constant[level*pos_dim + i-1]  ) * scale_factor_constant[level*pos_dim + i-1];
        elevated[i] = sm - i * cf;
        sm += cf;
    }
    elevated[0] = sm;

    int rem0[pos_dim+1];
    int rank[pos_dim+1]{0};




    // Find the closest 0-colored simplex through rounding
    // greedily search for the closest zero-colored lattice point
    int sum = 0;
    #pragma unroll
    for (int i = 0; i <= pos_dim; i++) {
        float v = elevated[i] * (1.0f / (pos_dim + 1));
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
        float delta = (elevated[i] - rem0[i]) * (1.0f / (pos_dim + 1));
        barycentric[pos_dim - rank[i]] += delta;
        barycentric[pos_dim + 1 - rank[i]] -= delta;
    }
    // Wrap around
    barycentric[0] += 1.0f + barycentric[pos_dim + 1];





    //here we accumulate the values and the homogeneous term
    float2 val_hom_vec=make_float2(0.0f, 0.0f);

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
        //     //tranposed
        //     splatting_indices[level][idx][remainder]=idx_val;
        //     splatting_weights[level][idx][remainder]=barycentric[remainder] * w_lvl;
        // }
        

        //if the vertex exists accumulate its value weighted by the barycentric weight (accumulates also the homogeneous coordinate)
        float w= barycentric[remainder] * w_lvl;

      

        //vectorized loads 
        // float* fv=&lattice_values_monolithic[level][idx_val][0];
        // float2 new_val=reinterpret_cast<float2*>( fv )[0];
        // val_hom_vec.x = val_hom_vec.x + new_val.x*w;
        // val_hom_vec.y = val_hom_vec.y + new_val.y*w;
        float* ptr_base=lattice_values_monolithic.data(); //tensor is nr_levels x capacity x nr_feat
        float* ptr_value=ptr_base+  idx_val*lattice_values_monolithic.stride(1) + level*lattice_values_monolithic.stride(0);
        float2 new_val=reinterpret_cast<float2*>( ptr_value )[0];
        val_hom_vec.x = val_hom_vec.x + new_val.x*w;
        val_hom_vec.y = val_hom_vec.y + new_val.y*w;

    }

   

    sliced_values_monolithic[level][0][idx]=val_hom_vec.x;
    sliced_values_monolithic[level][1][idx]=val_hom_vec.y;

    // val x res x nr_positions
    // sliced_values_monolithic[0][level][idx]=val_hom_vec.x;
    // sliced_values_monolithic[1][level][idx]=val_hom_vec.y;


}


template<int pos_dim, int val_dim>
__global__ void 
__launch_bounds__(BLOCK_SIZE_BACK) //since the block size is known at compile time we can specify it to the kernel and therefore cuda doesnt need to use heuristics based on code complexity to minimize registry usage
backward_gpu(
    const int nr_positions,
    const int lattice_capacity,
    const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> lattice_values_monolithic,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> positions,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> scale_factor,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> random_shift_monolithic,
    const torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> anneal_window,
    const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> grad_sliced_values_monolithic,
    torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> lattice_values_monolithic_grad,
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



    //default
    float2 grad_sliced_val_cur;
    grad_sliced_val_cur.x=grad_sliced_values_monolithic[level][0][idx];
    grad_sliced_val_cur.y=grad_sliced_values_monolithic[level][1][idx];







    float elevated[pos_dim + 1];


    float sm = 0;
    #pragma unroll
    for (int i = pos_dim; i > 0; i--) {
        // float cf = (positions[idx][i-1] +random_shift_monolithic[level][i-1]  ) * scale_factor[level][i - 1];
        float cf = (positions[idx][i-1] +random_shift_constant[level*pos_dim + i-1]  ) * scale_factor_constant[level*pos_dim + i-1];
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

           
            //if the vertex exists accumulate its value weighted by the barycentric weight (accumulates also the homogeneous coordinate)
            // #pragma unroll
            // for (int j = 0; j < val_dim ; j++){
                // float weighted_grad=grad_sliced_val_cur[j]*w;
                // atomicAdd(&lattice_values_monolithic_grad[level][idx_val][j], weighted_grad  );
            // }
            atomicAdd(&lattice_values_monolithic_grad[level][idx_val][0], grad_sliced_val_cur.x*w  );
            atomicAdd(&lattice_values_monolithic_grad[level][idx_val][1], grad_sliced_val_cur.y*w  );

        
        }
    }




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
            dL_dbarycentric[remainder]+=val_lattice_vertex.x*w_lvl   * grad_sliced_val_cur.x;
            dL_dbarycentric[remainder]+=val_lattice_vertex.y*w_lvl   * grad_sliced_val_cur.y;

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
        float dL_dPos[pos_dim]{0};
        //I unrolles the loop that computes E from P and I got some local derivatives like 
        //dEx/dPx=Sx  dEx/dPy=Sy
        //dEy/dPx=-Sx  dEy/dPy=Sy  dEy/dPz=Sz
        //dEz/dPy=-2Sy  dEz/dPz=Sz
        //dEw/dPz=-3Sz
        //So we just accumulate these values inot dL_dPos
        //x
        // dL_dPos[0]= dL_delevated[0]* scale_factor[level][0] +  
        //             dL_delevated[1]* (-scale_factor[level][0]);
        // //y
        // dL_dPos[1]= dL_delevated[0]* scale_factor[level][1] +  
        //             dL_delevated[1]* scale_factor[level][1] +
        //             dL_delevated[2]* (-2*scale_factor[level][1]);
        // //z
        // dL_dPos[2]= dL_delevated[0]* scale_factor[level][2] + 
        //             dL_delevated[1]* scale_factor[level][2] +
        //             dL_delevated[2]* scale_factor[level][2] +
        //             dL_delevated[3]* (-3*scale_factor[level][2]);
        //do it in a loop so as to support various pos_dims
        for(int i=0; i<pos_dim; i++){
            #pragma unroll
            for(int j=0; j<=i; j++){
                dL_dPos[i]+=dL_delevated[j]*scale_factor[level][i];
            }
        }
        #pragma unroll
        for(int i=0; i<pos_dim; i++){
            dL_dPos[i]-=dL_delevated[i+1] * scale_factor[level][i] * (i+1);
        }
        // if(debug) printf("dL_dPos[0] %f, dL_dPos[1] %f, dL_dPos[2] %f\n", dL_dPos[0], dL_dPos[1], dL_dPos[2]);
        //finish
        // printf("dL_dPos[0] %f \n",dL_dPos[0]);
        // atomicAdd(&positions_grad[idx][0], dL_dPos[0]  );
        // atomicAdd(&positions_grad[idx][1], dL_dPos[1]  );
        // atomicAdd(&positions_grad[idx][2], dL_dPos[2]  );
        #pragma unroll
        for(int i=0; i<pos_dim; i++){
            atomicAdd(&positions_grad[idx][i], dL_dPos[i]  );
        }
        //Cannot be done like this because the sums into the positions grad may come from multiple levels so they need to be atomic
        // positions_grad[idx][0]=dL_dPos[0];
        // positions_grad[idx][1]=dL_dPos[1];
        // positions_grad[idx][2]=dL_dPos[2];

        // positions_grad[level][idx][0]=dL_dPos[0];
        // positions_grad[level][idx][1]=dL_dPos[1];
        // positions_grad[level][idx][2]=dL_dPos[2];
        // #pragma unroll
        // for(int i=0; i<pos_dim; i++){
        //     positions_grad[level][idx][i]=dL_dPos[i];
        // }

    }

   
   


}

template<int pos_dim, int val_dim>
__global__ void 
__launch_bounds__(BLOCK_SIZE_BACK) //since the block size is known at compile time we can specify it to the kernel and therefore cuda doesnt need to use heuristics based on code complexity to minimize registry usage
backward_gpu_only_pos(
    const int nr_positions,
    const int lattice_capacity,
    const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> lattice_values_monolithic,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> positions,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> scale_factor,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> random_shift_monolithic,
    const torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> anneal_window,
    const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> grad_sliced_values_monolithic,
    torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> lattice_values_monolithic_grad,
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



    //default
    float2 grad_sliced_val_cur;
    grad_sliced_val_cur.x=grad_sliced_values_monolithic[level][0][idx];
    grad_sliced_val_cur.y=grad_sliced_values_monolithic[level][1][idx];







    float elevated[pos_dim + 1];


    float sm = 0;
    #pragma unroll
    for (int i = pos_dim; i > 0; i--) {
        // float cf = (positions[idx][i-1] +random_shift_monolithic[level][i-1]  ) * scale_factor[level][i - 1];
        float cf = (positions[idx][i-1] +random_shift_constant[level*pos_dim + i-1]  ) * scale_factor_constant[level*pos_dim + i-1];
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



    // float barycentric[pos_dim + 2]{0};
    // // Compute the barycentric coordinates (p.10 in [Adams etal 2010])
    // #pragma unroll
    // for (int i = 0; i <= pos_dim; i++) {
    //     float delta = (elevated[i] - rem0[i]) * (1.0 / (pos_dim + 1));
    //     barycentric[pos_dim - rank[i]] += delta;
    //     barycentric[pos_dim + 1 - rank[i]] -= delta;
    // }
    // // Wrap around
    // barycentric[0] += 1.0 + barycentric[pos_dim + 1];


    float w_lvl= anneal_window[level];


    int key[pos_dim];
    




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
            dL_dbarycentric[remainder]+=val_lattice_vertex.x*w_lvl   * grad_sliced_val_cur.x;
            dL_dbarycentric[remainder]+=val_lattice_vertex.y*w_lvl   * grad_sliced_val_cur.y;

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
        float dL_dPos[pos_dim]{0};
        //I unrolles the loop that computes E from P and I got some local derivatives like 
        //dEx/dPx=Sx  dEx/dPy=Sy
        //dEy/dPx=-Sx  dEy/dPy=Sy  dEy/dPz=Sz
        //dEz/dPy=-2Sy  dEz/dPz=Sz
        //dEw/dPz=-3Sz
        //So we just accumulate these values inot dL_dPos
        //x
        // dL_dPos[0]= dL_delevated[0]* scale_factor[level][0] +  
        //             dL_delevated[1]* (-scale_factor[level][0]);
        // //y
        // dL_dPos[1]= dL_delevated[0]* scale_factor[level][1] +  
        //             dL_delevated[1]* scale_factor[level][1] +
        //             dL_delevated[2]* (-2*scale_factor[level][1]);
        // //z
        // dL_dPos[2]= dL_delevated[0]* scale_factor[level][2] + 
        //             dL_delevated[1]* scale_factor[level][2] +
        //             dL_delevated[2]* scale_factor[level][2] +
        //             dL_delevated[3]* (-3*scale_factor[level][2]);
        //do it in a loop so as to support various pos_dims
        for(int i=0; i<pos_dim; i++){
            #pragma unroll
            for(int j=0; j<=i; j++){
                // dL_dPos[i]+=dL_delevated[j]*scale_factor[level][i];
                dL_dPos[i]+=dL_delevated[j]*scale_factor_constant[level*pos_dim + i];
            }
        }
        #pragma unroll
        for(int i=0; i<pos_dim; i++){
            dL_dPos[i]-=dL_delevated[i+1] * scale_factor_constant[level*pos_dim + i] * (i+1);
        }
        // if(debug) printf("dL_dPos[0] %f, dL_dPos[1] %f, dL_dPos[2] %f\n", dL_dPos[0], dL_dPos[1], dL_dPos[2]);
        //finish
        // printf("dL_dPos[0] %f \n",dL_dPos[0]);
        // atomicAdd(&positions_grad[idx][0], dL_dPos[0]  );
        // atomicAdd(&positions_grad[idx][1], dL_dPos[1]  );
        // atomicAdd(&positions_grad[idx][2], dL_dPos[2]  );
        #pragma unroll
        for(int i=0; i<pos_dim; i++){
            atomicAdd(&positions_grad[idx][i], dL_dPos[i]  );
        }
        //Cannot be done like this because the sums into the positions grad may come from multiple levels so they need to be atomic
        // positions_grad[idx][0]=dL_dPos[0];
        // positions_grad[idx][1]=dL_dPos[1];
        // positions_grad[idx][2]=dL_dPos[2];

        // positions_grad[level][idx][0]=dL_dPos[0];
        // positions_grad[level][idx][1]=dL_dPos[1];
        // positions_grad[level][idx][2]=dL_dPos[2];
        // #pragma unroll
        // for(int i=0; i<pos_dim; i++){
        //     positions_grad[level][idx][i]=dL_dPos[i];
        // }

    }

   
   


}



//double back
template<int pos_dim, int val_dim>
__global__ void 
__launch_bounds__(BLOCK_SIZE_DOUBLE_BACK) //since the block size is known at compile time we can specify it to the kernel and therefore cuda doesnt need to use heuristics based on code complexity to minimize registry usage
double_backward_from_positions_gpu(
    const int nr_positions,
    const int lattice_capacity,
    const int nr_resolutions,
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
    torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> lattice_values_monolithic_grad
    ){


    //values_vertices refers to the values that the lattice had in the forward pass. it has size m_hash_table_capcity x (val_dim+1)
    //grad_sliced_values is the gradient of the loss with respect to the sliced out values which has size nr_positions x val_dim
    const int idx = blockIdx.x * blockDim.x + threadIdx.x; // each thread will deal with one position
    if(idx >= nr_positions){
        return;
    }

    const uint32_t level = blockIdx.y; // <- the level is the same for all threads

    if(level>=nr_resolutions){
        //we are in one of the extra resolutions so we just write zero in the grad sliced grad
        grad_grad_sliced_values_monolithic[level][0][idx]=0;
        grad_grad_sliced_values_monolithic[level][1][idx]=0;
        return;
    }



    float elevated[pos_dim + 1];


    float sm = 0;
    #pragma unroll
    for (int i = pos_dim; i > 0; i--) {
        // float cf = (positions[idx][i-1] +random_shift_monolithic[level][i-1]  ) * scale_factor[level][i - 1];
        float cf = (positions[idx][i-1] +random_shift_constant[level*pos_dim + i-1]  ) * scale_factor_constant[level*pos_dim + i-1];
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
    #pragma unroll
    for (int j = 0; j < val_dim; j++) {
        grad_sliced_val_cur[j]=grad_sliced_values_monolithic[level][j][idx];
    }

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
    //-------hardocded for 3 positions----------
    // dL_delevated[0] =   grad_p_cur[0] * scale_factor[level][0] + 
    //                     grad_p_cur[1] * scale_factor[level][1] +
    //                     grad_p_cur[2] * scale_factor[level][2];
    // dL_delevated[1] =   grad_p_cur[0] * (-scale_factor[level][0]) + 
    //                     grad_p_cur[1] * scale_factor[level][1] +
    //                     grad_p_cur[2] * scale_factor[level][2];
    // dL_delevated[2] =   grad_p_cur[1] * (-2*scale_factor[level][1]) +
    //                     grad_p_cur[2] * scale_factor[level][2];
    // dL_delevated[3] =   grad_p_cur[2] * (-3*scale_factor[level][2]);
    //------doing it so that it support all pos_dims--------
    //in the forward pass we do:
    // for(int i=0; i<pos_dim; i++){
    //     for(int j=0; j<=i; j++){
    //         dL_dPos[i]+=dL_delevated[j]*scale_factor[level][i];
    //     }
    // }
    // for(int i=0; i<pos_dim; i++){
    //     dL_dPos[i]-=dL_delevated[i+1] * scale_factor[level][i] * (i+1);
    // }
    // so the gradient from grad_p_cur[i] will go into each elevated <= i. Afterwards we have another loop which passes the gradient from grad_p_cur[i] into elevated[i+1]
    for(int i=0; i<pos_dim; i++){
        // float grad=grad_p_cur[i]*scale_factor[level][i];
        float grad=grad_p_cur[i]*scale_factor_constant[level*pos_dim + i];
        #pragma unroll
        for(int j=0; j<=i; j++){
            dL_delevated[j]+=grad;
        }
    }
    #pragma unroll
    for(int i=0; i<pos_dim; i++){
        dL_delevated[i+1]-=grad_p_cur[i] * scale_factor_constant[level*pos_dim + i] * (i+1);
    }


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
        // atomicAdd(&lattice_values_monolithic_grad[level][0][idx_val], dL_dbarycentric[remainder]* w_lvl * grad_sliced_val_cur[0]  );
        // atomicAdd(&lattice_values_monolithic_grad[level][1][idx_val], dL_dbarycentric[remainder]* w_lvl * grad_sliced_val_cur[1]  );


        // push gradient into grad_sliced_val_cur
        // grad_sliced_val_cur add towards all the barycentric coord, so in the backward pass the gradient from b0 to all the grad_sliced_val_cur
        grad_grad_sliced_val_cur[0]+=dL_dbarycentric[remainder]* w_lvl * val_lattice_vertex.x;
        grad_grad_sliced_val_cur[1]+=dL_dbarycentric[remainder]* w_lvl * val_lattice_vertex.y;
    }
    //finish the accumulation of grad_grad_sliced
    // atomicAdd(&grad_grad_sliced_values_monolithic[level][0][idx], grad_grad_sliced_val_cur[0]  );
    // atomicAdd(&grad_grad_sliced_values_monolithic[level][1][idx], grad_grad_sliced_val_cur[1]  );
    grad_grad_sliced_values_monolithic[level][0][idx]=grad_grad_sliced_val_cur[0];
    grad_grad_sliced_values_monolithic[level][1][idx]=grad_grad_sliced_val_cur[1];

}



//double back
template<int pos_dim, int val_dim>
__global__ void 
__launch_bounds__(BLOCK_SIZE_DOUBLE_BACK) //since the block size is known at compile time we can specify it to the kernel and therefore cuda doesnt need to use heuristics based on code complexity to minimize registry usage
double_backward_from_positions_gpu_1(
    const int nr_positions,
    const int lattice_capacity,
    const int nr_resolutions,
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
    torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> lattice_values_monolithic_grad
    ){


    //values_vertices refers to the values that the lattice had in the forward pass. it has size m_hash_table_capcity x (val_dim+1)
    //grad_sliced_values is the gradient of the loss with respect to the sliced out values which has size nr_positions x val_dim
    const int idx = blockIdx.x * blockDim.x + threadIdx.x; // each thread will deal with one position
    if(idx >= nr_positions){
        return;
    }

    const uint32_t level = blockIdx.y; // <- the level is the same for all threads

    if(level>=nr_resolutions){
        //we are in one of the extra resolutions so we just write zero in the grad sliced grad
        // grad_grad_sliced_values_monolithic[level][0][idx]=0;
        // grad_grad_sliced_values_monolithic[level][1][idx]=0;
        // return;
    }



    float elevated[pos_dim + 1];


    float sm = 0;
    #pragma unroll
    for (int i = pos_dim; i > 0; i--) {
        // float cf = (positions[idx][i-1] +random_shift_monolithic[level][i-1]  ) * scale_factor[level][i - 1];
        float cf = (positions[idx][i-1] +random_shift_constant[level*pos_dim + i-1]  ) * scale_factor_constant[level*pos_dim + i-1];
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
    #pragma unroll
    for (int j = 0; j < val_dim; j++) {
        grad_sliced_val_cur[j]=grad_sliced_values_monolithic[level][j][idx];
    }

    // //get eh gradient at the curent position
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
    //-------hardocded for 3 positions----------
    // dL_delevated[0] =   grad_p_cur[0] * scale_factor[level][0] + 
    //                     grad_p_cur[1] * scale_factor[level][1] +
    //                     grad_p_cur[2] * scale_factor[level][2];
    // dL_delevated[1] =   grad_p_cur[0] * (-scale_factor[level][0]) + 
    //                     grad_p_cur[1] * scale_factor[level][1] +
    //                     grad_p_cur[2] * scale_factor[level][2];
    // dL_delevated[2] =   grad_p_cur[1] * (-2*scale_factor[level][1]) +
    //                     grad_p_cur[2] * scale_factor[level][2];
    // dL_delevated[3] =   grad_p_cur[2] * (-3*scale_factor[level][2]);
    //------doing it so that it support all pos_dims--------
    //in the forward pass we do:
    // for(int i=0; i<pos_dim; i++){
    //     for(int j=0; j<=i; j++){
    //         dL_dPos[i]+=dL_delevated[j]*scale_factor[level][i];
    //     }
    // }
    // for(int i=0; i<pos_dim; i++){
    //     dL_dPos[i]-=dL_delevated[i+1] * scale_factor[level][i] * (i+1);
    // }
    // so the gradient from grad_p_cur[i] will go into each elevated <= i. Afterwards we have another loop which passes the gradient from grad_p_cur[i] into elevated[i+1]
    for(int i=0; i<pos_dim; i++){
        // float grad=grad_p_cur[i]*scale_factor[level][i];
        float grad=grad_p_cur[i]*scale_factor_constant[level*pos_dim + i];
        #pragma unroll
        for(int j=0; j<=i; j++){
            dL_delevated[j]+=grad;
        }
    }
    #pragma unroll
    for(int i=0; i<pos_dim; i++){
        // dL_delevated[i+1]-=grad_p_cur[i] * scale_factor[level][i] * (i+1);
        dL_delevated[i+1]-=grad_p_cur[i] * scale_factor_constant[level*pos_dim + i] * (i+1);
    }
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
    #pragma unroll
    for (int i = 0; i <= pos_dim; i++) {
        dL_dbarycentric[pos_dim - rank[i]] += dL_delevated[i]* (1.0 / (pos_dim + 1));
        dL_dbarycentric[pos_dim + 1 - rank[i]] -= dL_delevated[i]* (1.0 / (pos_dim + 1));
    }
    // dL_dbarycentric[0]=dL_dbarycentric[0]*2;
    dL_dbarycentric[0] += dL_dbarycentric[pos_dim + 1];
    //push gradient into values_lattice and grad_sliced
    // float grad_grad_sliced_val_cur[val_dim]{0};
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
        // const float* fv=&lattice_values_monolithic[level][idx_val][0];
        // const float2 val_lattice_vertex=reinterpret_cast<const float2*>( fv )[0];
        //add to the dL_d_barycentric
        // dL_dbarycentric[remainder]+=val_lattice_vertex.x*w_lvl   * grad_sliced_val_cur[0];
        // dL_dbarycentric[remainder]+=val_lattice_vertex.y*w_lvl   * grad_sliced_val_cur[1];
        // lattice_values_monolithic_grad[level][idx_val][0] += dL_dbarycentric[remainder]* w_lvl * grad_sliced_val_cur[0];
        // lattice_values_monolithic_grad[level][idx_val][1] += dL_dbarycentric[remainder]* w_lvl * grad_sliced_val_cur[1];

        // atomicAdd(&lattice_values_monolithic_grad[level][idx_val][0], dL_dbarycentric[remainder]* w_lvl * grad_sliced_val_cur[0]  );
        // atomicAdd(&lattice_values_monolithic_grad[level][idx_val][1], dL_dbarycentric[remainder]* w_lvl * grad_sliced_val_cur[1]  );
        atomicAdd(&lattice_values_monolithic_grad[level][0][idx_val], dL_dbarycentric[remainder]* w_lvl * grad_sliced_val_cur[0]  );
        atomicAdd(&lattice_values_monolithic_grad[level][1][idx_val], dL_dbarycentric[remainder]* w_lvl * grad_sliced_val_cur[1]  );


        // push gradient into grad_sliced_val_cur
        // grad_sliced_val_cur add towards all the barycentric coord, so in the backward pass the gradient from b0 to all the grad_sliced_val_cur
        // grad_grad_sliced_val_cur[0]+=dL_dbarycentric[remainder]* w_lvl * val_lattice_vertex.x;
        // grad_grad_sliced_val_cur[1]+=dL_dbarycentric[remainder]* w_lvl * val_lattice_vertex.y;
    }
    //finish the accumulation of grad_grad_sliced
    // atomicAdd(&grad_grad_sliced_values_monolithic[level][0][idx], grad_grad_sliced_val_cur[0]  );
    // atomicAdd(&grad_grad_sliced_values_monolithic[level][1][idx], grad_grad_sliced_val_cur[1]  );

}

//double back
template<int pos_dim, int val_dim>
__global__ void 
__launch_bounds__(BLOCK_SIZE_DOUBLE_BACK) //since the block size is known at compile time we can specify it to the kernel and therefore cuda doesnt need to use heuristics based on code complexity to minimize registry usage
double_backward_from_positions_gpu_2(
    const int nr_positions,
    const int lattice_capacity,
    const int nr_resolutions,
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
    torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> lattice_values_monolithic_grad
    ){


    //values_vertices refers to the values that the lattice had in the forward pass. it has size m_hash_table_capcity x (val_dim+1)
    //grad_sliced_values is the gradient of the loss with respect to the sliced out values which has size nr_positions x val_dim
    const int idx = blockIdx.x * blockDim.x + threadIdx.x; // each thread will deal with one position
    if(idx >= nr_positions){
        return;
    }

    const uint32_t level = blockIdx.y; // <- the level is the same for all threads

    if(level>=nr_resolutions){
        //we are in one of the extra resolutions so we just write zero in the grad sliced grad
        grad_grad_sliced_values_monolithic[level][0][idx]=0;
        grad_grad_sliced_values_monolithic[level][1][idx]=0;
        return;
    }



    float elevated[pos_dim + 1];


    float sm = 0;
    #pragma unroll
    for (int i = pos_dim; i > 0; i--) {
        // float cf = (positions[idx][i-1] +random_shift_monolithic[level][i-1]  ) * scale_factor[level][i - 1];
        float cf = (positions[idx][i-1] +random_shift_constant[level*pos_dim + i-1]  ) * scale_factor_constant[level*pos_dim + i-1];
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
    //-------hardocded for 3 positions----------
    // dL_delevated[0] =   grad_p_cur[0] * scale_factor[level][0] + 
    //                     grad_p_cur[1] * scale_factor[level][1] +
    //                     grad_p_cur[2] * scale_factor[level][2];
    // dL_delevated[1] =   grad_p_cur[0] * (-scale_factor[level][0]) + 
    //                     grad_p_cur[1] * scale_factor[level][1] +
    //                     grad_p_cur[2] * scale_factor[level][2];
    // dL_delevated[2] =   grad_p_cur[1] * (-2*scale_factor[level][1]) +
    //                     grad_p_cur[2] * scale_factor[level][2];
    // dL_delevated[3] =   grad_p_cur[2] * (-3*scale_factor[level][2]);
    //------doing it so that it support all pos_dims--------
    //in the forward pass we do:
    // for(int i=0; i<pos_dim; i++){
    //     for(int j=0; j<=i; j++){
    //         dL_dPos[i]+=dL_delevated[j]*scale_factor[level][i];
    //     }
    // }
    // for(int i=0; i<pos_dim; i++){
    //     dL_dPos[i]-=dL_delevated[i+1] * scale_factor[level][i] * (i+1);
    // }
    // so the gradient from grad_p_cur[i] will go into each elevated <= i. Afterwards we have another loop which passes the gradient from grad_p_cur[i] into elevated[i+1]
    for(int i=0; i<pos_dim; i++){
        // float grad=grad_p_cur[i]*scale_factor[level][i];
        float grad=grad_p_cur[i]*scale_factor_constant[level*pos_dim + i];
        #pragma unroll
        for(int j=0; j<=i; j++){
            dL_delevated[j]+=grad;
        }
    }
    #pragma unroll
    for(int i=0; i<pos_dim; i++){
        // dL_delevated[i+1]-=grad_p_cur[i] * scale_factor[level][i] * (i+1);
        dL_delevated[i+1]-=grad_p_cur[i] * scale_factor_constant[level*pos_dim + i] * (i+1);
    }
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
    #pragma unroll
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
        // atomicAdd(&lattice_values_monolithic_grad[level][idx_val][0], dL_dbarycentric[remainder]* w_lvl * grad_sliced_val_cur[0]  );
        // atomicAdd(&lattice_values_monolithic_grad[level][idx_val][1], dL_dbarycentric[remainder]* w_lvl * grad_sliced_val_cur[1]  );


        // push gradient into grad_sliced_val_cur
        // grad_sliced_val_cur add towards all the barycentric coord, so in the backward pass the gradient from b0 to all the grad_sliced_val_cur
        grad_grad_sliced_val_cur[0]+=dL_dbarycentric[remainder]* w_lvl * val_lattice_vertex.x;
        grad_grad_sliced_val_cur[1]+=dL_dbarycentric[remainder]* w_lvl * val_lattice_vertex.y;
    }
    //finish the accumulation of grad_grad_sliced
    // atomicAdd(&grad_grad_sliced_values_monolithic[level][0][idx], grad_grad_sliced_val_cur[0]  );
    // atomicAdd(&grad_grad_sliced_values_monolithic[level][1][idx], grad_grad_sliced_val_cur[1]  );
    grad_grad_sliced_values_monolithic[level][0][idx]=grad_grad_sliced_val_cur[0];
    grad_grad_sliced_values_monolithic[level][1][idx]=grad_grad_sliced_val_cur[1];

}



