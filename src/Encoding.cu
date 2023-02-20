#include "permutohedral_encoding/Encoding.cuh"

//c++
#include <string>


//my stuff
#include "permutohedral_encoding/EncodingGPU.cuh"


using torch::Tensor;


template <typename T>
T div_round_up(T val, T divisor) {
	return (val + divisor - 1) / divisor;
}



template<uint32_t POS_DIM, uint32_t NR_FEAT_PER_LEVEL>
Encoding<POS_DIM, NR_FEAT_PER_LEVEL>::Encoding(const EncodingFixedParams& fixed_params):
    m_fixed_params(fixed_params)
    {
}


template<uint32_t POS_DIM, uint32_t NR_FEAT_PER_LEVEL>
Encoding<POS_DIM, NR_FEAT_PER_LEVEL>::~Encoding(){
}



template<uint32_t POS_DIM, uint32_t NR_FEAT_PER_LEVEL>
void Encoding<POS_DIM, NR_FEAT_PER_LEVEL>::check_positions(const torch::Tensor& positions_raw){
    CHECK(positions_raw.is_cuda()) << "positions should be in GPU memory. Please call .cuda() on the tensor";
    CHECK(positions_raw.scalar_type()==at::kFloat) << "positions should be of type float";
    CHECK(positions_raw.dim()==2) << "positions should have dim 2 correspondin to HW. However it has sizes" << positions_raw.sizes();
    int pos_dim=positions_raw.size(1);
    // CHECK(m_sigmas.size()==pos_dim) <<"One must set sigmas for each dimension of the positions. Use set_sigmas. m_sigmas is " << m_sigmas.size() << " m_pos dim is " <<pos_dim;
    // CHECK(m_expected_pos_dim==positions_raw.size(1)) << "The expected pos dim is " << m_expected_pos_dim << " whole the input points have pos_dim " << positions_raw.size(1);
    CHECK(positions_raw.size(0)!=0) << "Why do we have 0 points";
    CHECK(positions_raw.size(1)!=0) << "Why do we have dimension 0 for the points";
    // CHECK(positions_raw.is_contiguous()) << "Positions raw is not contiguous. Please call .contiguous() on it";
    // CHECK(pos_dim==m_expected_position_dimensions) << "The pos dim should be the same as the expected positions dimensions given by the sigmas. Pos dim is " << pos_dim << " m_expected_position_dimensions " << m_expected_position_dimensions;
}
template<uint32_t POS_DIM, uint32_t NR_FEAT_PER_LEVEL>
void Encoding<POS_DIM, NR_FEAT_PER_LEVEL>::check_values(const torch::Tensor& values){
    CHECK(values.is_cuda()) << "lattice values should be in GPU memory. Please call .cuda() on the tensor";
    CHECK(values.scalar_type()==at::kFloat) << "values should be of type float";
    CHECK(values.dim()==2) << "values should have dim 2 correspondin to HW. However it has sizes" << values.sizes();
    CHECK(values.is_contiguous()) << "Values is not contiguous. Please call .contiguous() on it";
}
template<uint32_t POS_DIM, uint32_t NR_FEAT_PER_LEVEL>
void Encoding<POS_DIM, NR_FEAT_PER_LEVEL>::check_positions_and_values(const torch::Tensor& positions_raw, const torch::Tensor& values){
    //check input
    CHECK(positions_raw.size(0)==values.size(0)) << "Sizes of positions and values should match. Meaning that that there should be a value for each position. Positions_raw has sizes "<<positions_raw.sizes() << " and the values has size " << values.sizes();
    check_positions(positions_raw);
    check_values(positions_raw);
}







template<uint32_t POS_DIM, uint32_t NR_FEAT_PER_LEVEL>
torch::Tensor Encoding<POS_DIM, NR_FEAT_PER_LEVEL>::forward(const EncodingInput& input){

    check_positions(input.m_positions_raw); 
    int nr_positions=input.m_positions_raw.size(0);
    int pos_dim=input.m_positions_raw.size(1);
    int nr_resolutions=input.m_lattice_values.size(0);
    int lattice_capacity=input.m_lattice_values.size(1);
    int val_dim=input.m_lattice_values.size(2);
    CHECK(m_fixed_params.m_random_shift_per_level.size(0)==nr_resolutions ) <<"Random shift should have the first dimension the same as the nr of resolutions";
    CHECK(m_fixed_params.m_random_shift_per_level.size(1)==pos_dim ) <<"Random shift should have the second dimension the same as the pos dim";
    //check the anneal window
    CHECK(input.m_anneal_window.size(0)==nr_resolutions ) <<"anneal_window should have the first dimension the same as the nr of resolutions";



    //if we concat also the points, we add a series of extra resolutions to contain those points
    int nr_resolutions_extra=0;
    if (m_fixed_params.m_concat_points){
        nr_resolutions_extra=std::ceil(float(pos_dim)/val_dim);
    }



    //initialize the output values 
    Tensor sliced_values_hom_tensor=torch::empty({nr_resolutions+nr_resolutions_extra, val_dim, nr_positions }, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0) );

  

    //try again with a monolithic kernel
    const dim3 blocks = { (unsigned int)div_round_up(nr_positions, BLOCK_SIZE), (unsigned int)(nr_resolutions+nr_resolutions_extra), 1 }; //the blocks are executed in order, first the blocks for the first resolution, then the second and so on
    
    forward_gpu<POS_DIM, NR_FEAT_PER_LEVEL><<<blocks, BLOCK_SIZE>>>(
        nr_positions, 
        lattice_capacity,
        nr_resolutions,
        nr_resolutions_extra,
        input.m_positions_raw.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
        input.m_lattice_values.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
        m_fixed_params.m_scale_factor.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
        m_fixed_params.m_random_shift_per_level.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
        input.m_anneal_window.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
        sliced_values_hom_tensor.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
        m_fixed_params.m_concat_points,
        m_fixed_params.m_points_scaling,
        input.m_require_lattice_values_grad,
        input.m_require_positions_grad
    );
   


    return sliced_values_hom_tensor;

}





template<uint32_t POS_DIM, uint32_t NR_FEAT_PER_LEVEL>
std::tuple<torch::Tensor, torch::Tensor> Encoding<POS_DIM, NR_FEAT_PER_LEVEL>::backward(const EncodingInput& input, torch::Tensor& grad_sliced_values_monolithic){


    check_positions(input.m_positions_raw); 
    int nr_positions=input.m_positions_raw.size(0);
    int pos_dim=input.m_positions_raw.size(1);
    int capacity=input.m_lattice_values.size(1);
    CHECK(grad_sliced_values_monolithic.dim()==3) <<"grad_sliced_values_monolithic should be nr_resolutions x val_dim x nr_positions, so it should have 3 dimensions. However it has "<< grad_sliced_values_monolithic.dim();
    CHECK(grad_sliced_values_monolithic.is_contiguous()) << "Grad sliced values needs to be contiguous. Please call .contiguous() on it";
    int nr_resolutions=grad_sliced_values_monolithic.size(0);
    int val_dim=grad_sliced_values_monolithic.size(1);
    CHECK(nr_positions==grad_sliced_values_monolithic.size(2)) << "The nr of positions should match between the input positions and the sliced values";
    CHECK(input.m_lattice_values.dim()==3) <<"grad_sliced_values_monolithic should be nr_resolutions x val_dim x nr_positions, so it should have 3 dimensions. However it has "<< input.m_lattice_values.dim();
    CHECK(input.m_lattice_values.is_contiguous()) <<"We assume that the lattice_values_monolithic are contiguous because in the cuda code we make a load of 2 float values at a time and that assumes that they are contiguous";
    
    

    //if we concat also the points, we add a series of extra resolutions to contain those points
    int nr_resolutions_extra=0;
    if (m_fixed_params.m_concat_points){
        nr_resolutions_extra=std::ceil(float(pos_dim)/val_dim);
        nr_resolutions=nr_resolutions-nr_resolutions_extra;
    }


   
    Tensor lattice_values_monolithic_grad; //dL/dLattiveValues
    if (input.m_require_lattice_values_grad){
        lattice_values_monolithic_grad=torch::zeros({ nr_resolutions, capacity, val_dim },  torch::dtype(torch::kFloat32).device(torch::kCUDA, 0)  );
    }else{
        lattice_values_monolithic_grad=torch::empty({ 1,1,1 },  torch::dtype(torch::kFloat32).device(torch::kCUDA, 0)  );
    }
    


    Tensor positions_grad; //dL/dPos
    if (input.m_require_positions_grad){
        positions_grad=torch::zeros({ nr_positions, pos_dim },  torch::dtype(torch::kFloat32).device(torch::kCUDA, 0)  );
    }else{
        positions_grad=torch::empty({ 1,1 },  torch::dtype(torch::kFloat32).device(torch::kCUDA, 0)  );
    }

   

    const dim3 blocks = { (unsigned int)div_round_up(nr_positions, BLOCK_SIZE_BACK), (unsigned int)nr_resolutions, 1 }; //the blocks are executed in order, first the blocks for the first resolution, then the second and so on

    backward_gpu<POS_DIM,NR_FEAT_PER_LEVEL><<<blocks, BLOCK_SIZE_BACK>>>(
        nr_positions,
        capacity, 
        input.m_lattice_values.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
        input.m_positions_raw.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
        m_fixed_params.m_scale_factor.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
        m_fixed_params.m_random_shift_per_level.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
        input.m_anneal_window.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
        grad_sliced_values_monolithic.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
        lattice_values_monolithic_grad.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
        positions_grad.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
        m_fixed_params.m_concat_points,
        input.m_require_lattice_values_grad,
        input.m_require_positions_grad
    );


    return std::make_tuple(lattice_values_monolithic_grad, positions_grad);

}



template<uint32_t POS_DIM, uint32_t NR_FEAT_PER_LEVEL>
std::tuple<torch::Tensor, torch::Tensor> Encoding<POS_DIM, NR_FEAT_PER_LEVEL>::double_backward_from_positions(const EncodingInput& input, const torch::Tensor& double_positions_grad, torch::Tensor& grad_sliced_values_monolithic){

    check_positions(input.m_positions_raw); 
    int nr_positions=input.m_positions_raw.size(0);
    int pos_dim=input.m_positions_raw.size(1);
    int capacity=input.m_lattice_values.size(1);
    CHECK(grad_sliced_values_monolithic.dim()==3) <<"grad_sliced_values_monolithic should be nr_resolutions x val_dim x nr_positions, so it should have 3 dimensions. However it has "<< grad_sliced_values_monolithic.dim();
    CHECK(grad_sliced_values_monolithic.is_contiguous()) << "Grad sliced values needs to be contiguous. Please call .contiguous() on it";
    int nr_resolutions=grad_sliced_values_monolithic.size(0);
    int val_dim=grad_sliced_values_monolithic.size(1);
    CHECK(nr_positions==grad_sliced_values_monolithic.size(2)) << "The nr of positions should match between the input positions and the sliced values";
    CHECK(input.m_lattice_values.dim()==3) <<"grad_sliced_values_monolithic should be nr_resolutions x val_dim x nr_positions, so it should have 3 dimensions. However it has "<< input.m_lattice_values.dim();
    CHECK(input.m_lattice_values.is_contiguous()) <<"We assume that the lattice_values_monolithic are contiguous because in the cuda code we make a load of 2 float values at a time and that assumes that they are contiguous";


    //if we concat also the points, we add a series of extra resolutions to contain those points
    int nr_resolutions_extra=0;
    if (m_fixed_params.m_concat_points){
        nr_resolutions_extra=std::ceil(float(pos_dim)/val_dim);
        nr_resolutions=nr_resolutions-nr_resolutions_extra;
    }

    

    // nr_resolutions x nr_lattice_vertices x nr_lattice_featues
    //dL/dLattiveValues
    Tensor lattice_values_monolithic_grad=torch::zeros({ nr_resolutions, capacity, val_dim },  torch::dtype(torch::kFloat32).device(torch::kCUDA, 0)  );

    Tensor grad_grad_sliced_values_monolithic = torch::zeros({ nr_resolutions+nr_resolutions_extra, val_dim, nr_positions },  torch::dtype(torch::kFloat32).device(torch::kCUDA, 0)  );
    

    


    const dim3 blocks = { (unsigned int)div_round_up(nr_positions, BLOCK_SIZE_BACK), (unsigned int)nr_resolutions, 1 }; //the blocks are executed in order, first the blocks for the first resolution, then the second and so on

   
    double_backward_from_positions_gpu<POS_DIM, NR_FEAT_PER_LEVEL><<<blocks, BLOCK_SIZE_BACK>>>(
        nr_positions,
        capacity, 
        double_positions_grad.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
        input.m_lattice_values.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
        input.m_positions_raw.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
        m_fixed_params.m_scale_factor.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
        m_fixed_params.m_random_shift_per_level.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
        input.m_anneal_window.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
        grad_sliced_values_monolithic.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
        m_fixed_params.m_concat_points,
        //output
        grad_grad_sliced_values_monolithic.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
        lattice_values_monolithic_grad.packed_accessor32<float,3,torch::RestrictPtrTraits>()
    );

    
   

    return std::make_tuple(lattice_values_monolithic_grad,  grad_grad_sliced_values_monolithic);

}






//explicit instantiation 
// https://stackoverflow.com/a/495056
// https://isocpp.org/wiki/faq/templates#separate-template-class-defn-from-decl
//for val 2
template class Encoding<2,2>;
template class Encoding<3,2>;
template class Encoding<4,2>;
template class Encoding<5,2>;
template class Encoding<6,2>;
//for val 4 
//TODO not implemented other values other than 2 because we assume we load only 2 floats in the CUDA kernels



