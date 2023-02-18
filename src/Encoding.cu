#include "permutohedral_encoding/Encoding.cuh"

//c++
#include <string>

// #include "UtilsPytorch.h" //contains torch so it has to be added BEFORE any other include because the other ones might include loguru which gets screwed up if torch was included before it
// #include "EasyCuda/UtilsCuda.h"
// #include "string_utils.h"

//my stuff
// #include "instant_ngp_2/HashTable.cuh"
#include "permutohedral_encoding/EncodingGPU.cuh"

// //jitify
// #define JITIFY_PRINT_INSTANTIATION 1
// #define JITIFY_PRINT_SOURCE 1
// #define JITIFY_PRINT_LOG 1
// #define JITIFY_PRINT_PTX 1
// #define JITIFY_PRINT_LAUNCH 1

//loguru
// #define LOGURU_REPLACE_GLOG 1
// #include <loguru.hpp> //needs to be added after torch.h otherwise loguru stops printing for some reason

// //configuru
// #define CONFIGURU_WITH_EIGEN 1
// #define CONFIGURU_IMPLICIT_CONVERSIONS 1
// #include <configuru.hpp>
// using namespace configuru;
// //Add this header after we add all cuda stuff because we need the profiler to have cudaDeviceSyncronize defined
// #define ENABLE_CUDA_PROFILING 1
// #include "Profiler.h" 

//boost
// #include <boost/filesystem.hpp>
// namespace fs = boost::filesystem;

// #include <cuda_fp16.h>

using torch::Tensor;
// using namespace radu::utils;


// int Lattice::m_expected_position_dimensions = -1;


template <typename T>
T div_round_up(T val, T divisor) {
	return (val + divisor - 1) / divisor;
}


//CPU code that calls the kernels
// Lattice::Lattice(const std::string config_file):
//     // m_impl( new LatticeGPU() ),
//     m_lvl(1)
//     // m_expected_position_dimensions(0)
//     {

//     // init_params(config_file);
//     VLOG(3) << "Creating lattice";

// }


Lattice::Lattice(const int pos_dim, const int capacity, const int nr_levels, const int nr_feat_per_level):
    m_expected_pos_dim(pos_dim),
    m_capacity(capacity),
    m_nr_levels(nr_levels),
    m_nr_feat_per_level(nr_feat_per_level)
    {

    // init_params(config_file);
    // VLOG(3) << "Creating lattice";

}


// Lattice::Lattice(const std::string config_file, const std::string name):
//     // m_impl( new LatticeGPU() ),
//     m_name(name),
//     m_lvl(1)
//     {

//     // init_params(config_file);


//     VLOG(3) << "Creating lattice: " <<name;

// }

// Lattice::Lattice(Lattice* other):
//     // m_impl( new LatticeGPU() ),
//     m_lvl(1)
//     {
//         m_lvl=other->m_lvl;
//         // m_pos_dim=other->m_pos_dim;
//         // m_val_dim=other->m_val_dim;
//         // m_hash_table_capacity=other->m_hash_table_capacity;
//         m_sigmas=other->m_sigmas;
//         m_sigmas_tensor=other->m_sigmas_tensor.clone(); //deep copy
//         // m_expected_position_dimensions=other->m_expected_position_dimensions;
//         // m_splatting_indices_tensor=other->m_splatting_indices_tensor; //shallow copy
//         // m_splatting_weights_tensor=other->m_splatting_weights_tensor; //shallow copy
//         // m_lattice_rowified=other->m_lattice_rowified; //shallow copy
//         m_positions=other->m_positions; //shallow copy
//         //hashtable
//         // m_hash_table=std::make_shared<HashTable>(other->hash_table()->capacity() );
//         // m_hash_table->m_capacity=other->m_hash_table_capacity; 
//         // m_hash_table->m_pos_dim=other->m_pos_dim;
//         //hashtable tensors shallow copy (just a pointer assignemtn so they use the same data in memory)
//         // m_hash_table->m_keys_tensor=other->m_hash_table->m_keys_tensor;
//         // m_hash_table->m_values_tensor=other->m_hash_table->m_values_tensor;
//         // m_hash_table->m_entries_tensor=other->m_hash_table->m_entries_tensor;
//         // m_hash_table->m_nr_filled_tensor=other->m_hash_table->m_nr_filled_tensor.clone(); //deep copy for this one as the new lattice may have different number of vertices
//         // m_hash_table->m_nr_filled=m_hash_table->m_nr_filled;
//         // m_hash_table->m_nr_filled_is_dirty=m_hash_table->m_nr_filled_is_dirty;
//         // m_hash_table->update_impl();

// }

Lattice::~Lattice(){
    // LOG(WARNING) << "Deleting lattice: " << m_name;
}

// void Lattice::init(const int val_dim){
//     m_hash_table->init(m_expected_position_dimensions, val_dim);
//     m_hash_table->to(torch::kCUDA);
// }

// void Lattice::clear(){
//     m_hash_table->clear(); 
// }

// void Lattice::clear_only_values(){
//     m_hash_table->clear_only_values(); 
// }

// void Lattice::init_params(const std::string config_file){
//     // Config cfg = configuru::parse_file(std::string(CMAKE_SOURCE_DIR)+"/config/"+config_file, CFG);
//     std::string config_file_abs;
//     if (fs::path(config_file).is_relative()){
//         config_file_abs=(fs::path(PROJECT_SOURCE_DIR) / config_file).string();
//     }else{
//         config_file_abs=config_file;
//     }
//     Config cfg = configuru::parse_file(config_file_abs, CFG);
//     Config lattice_config=cfg["lattice_gpu"];
//     int hash_table_capacity = lattice_config["hash_table_capacity"];
//     m_capacity=hash_table_capacity;
//     // m_hash_table=std::make_shared<HashTable>(hash_table_capacity);

//     int nr_sigmas=lattice_config["nr_sigmas"]; //nr of is sigma values we have. Each one affecting a different number of dimensions of the positions
//     for (int i=0; i < nr_sigmas; i++) {
//         std::string param_name="sigma_"+std::to_string(i);
//         std::string sigma_val_and_extent = (std::string)lattice_config[param_name];
//         std::vector<std::string> tokenized = radu::utils::split(sigma_val_and_extent, " ");
//         CHECK(tokenized.size()==2) << "For each sigma we must define its value and the extent(nr of dimensions it affects) in space separated string. So the nr of tokens split string should have would be 1. However the nr of tokens we have is" << tokenized.size();
//         std::pair<float, int> sigma_params = std::make_pair<float,int> (  std::stof(tokenized[0]), std::stof(tokenized[1]) );
//         m_sigmas_val_and_extent.push_back(sigma_params);
//     }
//     set_sigmas(m_sigmas_val_and_extent);


// }

// void Lattice::set_sigmas_from_string(std::string sigma_val_and_extent){
//     m_sigmas_val_and_extent.clear();
//     std::vector<std::string> tokenized = radu::utils::split(sigma_val_and_extent, " ");
//     CHECK(tokenized.size()==2) << "For each sigma we must define its value and the extent(nr of dimensions it affects) in space separated string. So the nr of tokens split string should have would be 1. However the nr of tokens we have is" << tokenized.size();
//     std::pair<float, int> sigma_params = std::make_pair<float,int> (  std::stof(tokenized[0]), std::stof(tokenized[1]) );
//     m_sigmas_val_and_extent.push_back(sigma_params);

//     set_sigmas(m_sigmas_val_and_extent);

// }


// void Lattice::set_sigmas(std::initializer_list<  std::pair<float, int> > sigmas_list){
//     m_sigmas.clear();
//     for(auto sigma_pair : sigmas_list){
//         float sigma=sigma_pair.first; //value of the sigma
//         int nr_dim=sigma_pair.second; //how many dimensions are affected by this sigma
//         for(int i=0; i < nr_dim; i++){
//             m_sigmas.push_back(sigma);
//         }
//     }
//     // m_expected_position_dimensions=m_sigmas.size();

//     m_sigmas_tensor=vec2tensor(m_sigmas);
//     m_sigmas_tensor=m_sigmas_tensor.to(torch::kFloat32);
// }

// void Lattice::set_sigmas(std::vector<  std::pair<float, int> > sigmas_list){
//     m_sigmas.clear();
//     for(auto sigma_pair : sigmas_list){
//         float sigma=sigma_pair.first; //value of the sigma
//         int nr_dim=sigma_pair.second; //how many dimensions are affected by this sigma
//         for(int i=0; i < nr_dim; i++){
//             m_sigmas.push_back(sigma);
//         }
//     }
//     // m_expected_position_dimensions=m_sigmas.size();

//     m_sigmas_tensor=vec2tensor(m_sigmas);
// }

void Lattice::check_positions(const torch::Tensor& positions_raw){
    CHECK(positions_raw.scalar_type()==at::kFloat) << "positions should be of type float";
    CHECK(positions_raw.dim()==2) << "positions should have dim 2 correspondin to HW. However it has sizes" << positions_raw.sizes();
    int pos_dim=positions_raw.size(1);
    // CHECK(m_sigmas.size()==pos_dim) <<"One must set sigmas for each dimension of the positions. Use set_sigmas. m_sigmas is " << m_sigmas.size() << " m_pos dim is " <<pos_dim;
    CHECK(m_expected_pos_dim==positions_raw.size(1)) << "The expected pos dim is " << m_expected_pos_dim << " whole the input points have pos_dim " << positions_raw.size(1);
    CHECK(positions_raw.size(0)!=0) << "Why do we have 0 points";
    CHECK(positions_raw.size(1)!=0) << "Why do we have dimension 0 for the points";
    // CHECK(positions_raw.is_contiguous()) << "Positions raw is not contiguous. Please call .contiguous() on it";
    // CHECK(pos_dim==m_expected_position_dimensions) << "The pos dim should be the same as the expected positions dimensions given by the sigmas. Pos dim is " << pos_dim << " m_expected_position_dimensions " << m_expected_position_dimensions;
}

void Lattice::check_positions_elevated(const torch::Tensor& positions_elevated){
    CHECK(positions_elevated.scalar_type()==at::kFloat) << "positions should be of type float";
    CHECK(positions_elevated.dim()==2) << "positions should have dim 2 correspondin to HW. However it has sizes" << positions_elevated.sizes();
    int pos_dim=positions_elevated.size(1)-1;
    CHECK(m_sigmas.size()==pos_dim) <<"One must set sigmas for each dimension of the positions. Use set_sigmas. m_sigmas is " << m_sigmas.size() << " m_pos dim is " <<pos_dim;
    CHECK(positions_elevated.is_contiguous()) << "Positions raw is not contiguous. Please call .contiguous() on it";
    // CHECK(pos_dim==m_expected_position_dimensions) << "The pos dim should be the same as the expected positions dimensions given by the sigmas. Pos dim is " << pos_dim << " m_expected_position_dimensions " << m_expected_position_dimensions;
}

void Lattice::check_values(const torch::Tensor& values){
    CHECK(values.scalar_type()==at::kFloat) << "values should be of type float";
    CHECK(values.dim()==2) << "values should have dim 2 correspondin to HW. However it has sizes" << values.sizes();
    CHECK(values.is_contiguous()) << "Values is not contiguous. Please call .contiguous() on it";
}
void Lattice::check_positions_and_values(const torch::Tensor& positions_raw, const torch::Tensor& values){
    //check input
    CHECK(positions_raw.size(0)==values.size(0)) << "Sizes of positions and values should match. Meaning that that there should be a value for each position. Positions_raw has sizes "<<positions_raw.sizes() << " and the values has size " << values.sizes();
    check_positions(positions_raw);
    check_values(positions_raw);
}





//forward
// std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> Lattice::slice_with_collisions_standalone_no_precomputation(torch::Tensor& positions_raw, const bool should_precompute_tensors_for_backward){
//     check_positions(positions_raw); 
//     CHECK(val_dim()>0) << "m_val_dim is 0 or lwoer. We have to splat something first so that we have values from where to slice. Val dim is " << val_dim();
//     int nr_positions=positions_raw.size(0);
//     int pos_dim=positions_raw.size(1);
//     CHECK(pos_dim==this->pos_dim()) << " The position dimension do not coreespond with the ones we used for creating the lattice";


//      //to cuda
//     // TIME_START("upload_cuda");
//     positions_raw=positions_raw.to("cuda");
//     m_sigmas_tensor=m_sigmas_tensor.to("cuda");
//     // TIME_END("upload_cuda");

//     // TIME_START("scale_by_sigma");
//     // VLOG(3) << "slice standalone scaling by a sigma of " << m_sigmas_tensor;
//     Tensor positions=positions_raw/m_sigmas_tensor;
//     // TIME_END("scale_by_sigma")

//     //initialize the output values to zero 
//     // TIME_START("create_tensors")
//     Tensor sliced_values_hom_tensor=torch::zeros({nr_positions, val_dim() }, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0) );
//     // Tensor sliced_values_hom_tensor=torch::zeros({ val_dim(), nr_positions }, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0) );

//     //recalculate the splatting indices and weight for the backward pass of the slice
//     Tensor splatting_indices_tensor;
//     Tensor splatting_weights_tensor;
//     if (should_precompute_tensors_for_backward){
//         splatting_indices_tensor = torch::empty({ (pos_dim+1)* nr_positions }, torch::dtype(torch::kInt32).device(torch::kCUDA, 0) );
//         splatting_weights_tensor = torch::empty({ (pos_dim+1)* nr_positions }, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0) );
//         splatting_indices_tensor.fill_(-1);
//         splatting_weights_tensor.fill_(-1);
//     }else{
//         splatting_indices_tensor = torch::empty({ 1 }, torch::dtype(torch::kInt32).device(torch::kCUDA, 0) );
//         splatting_weights_tensor = torch::empty({ 1 }, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0) );
//     }
//     m_hash_table->update_impl();

//     //transpose the values
//     int val_dim=m_hash_table->m_values_tensor.size(1);
//     // m_hash_table->m_values_tensor=m_hash_table->m_values_tensor.transpose(0,1).contiguous();;
//     // m_hash_table->update_impl();
//     // positions=positions.transpose(0,1).contiguous();


//     //this makes the access to these more coalesced so the slicing itself is faster but this transpose and making into a contiguous array just makes everything slower in the end so we disable it
//     //either way there is little difference in the slicing speed when the nr of features is low
//     // positions=positions.transpose(0,1).contiguous().transpose(0,1);
//     // sliced_values_hom_tensor=sliced_values_hom_tensor.transpose(0,1).contiguous().transpose(0,1);



//     // TIME_START("slice");
//     // m_impl->slice_with_collisions_standalone_no_precomputation( positions.data_ptr<float>(), sliced_values_hom_tensor.data_ptr<float>(), m_expected_position_dimensions, val_dim,  nr_positions, splatting_indices_tensor.data_ptr<int>(), splatting_weights_tensor.data_ptr<float>(), *(m_hash_table->m_impl) );

//     dim3 blocks((nr_positions - 1) / BLOCK_SIZE + 1, 1, 1);
//     dim3 blockSize(BLOCK_SIZE, 1, 1);
//     blockSize.y = 1;
//     // slice_with_collisions_no_precomputation<3><<<blocks, blockSize>>>(1, positions.data_ptr<float>(), sliced_values_hom_tensor.data_ptr<float>(), nr_positions, splatting_indices_tensor.data_ptr<int>(), splatting_weights_tensor.data_ptr<float>(), *(m_hash_table->m_impl) );

//     if (pos_dim==3){
//         if(val_dim==2){
//             slice_with_collisions_no_precomputation<3, 2><<<blocks, blockSize>>>(
//                 nr_positions, 
//                 positions.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
//                 sliced_values_hom_tensor.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
//                 splatting_indices_tensor.packed_accessor32<int,1,torch::RestrictPtrTraits>(),   
//                 splatting_weights_tensor.packed_accessor32<float,1,torch::RestrictPtrTraits>(),   
//                 should_precompute_tensors_for_backward,
//                 *(m_hash_table->m_impl),
//                 m_hash_table->m_values_tensor.packed_accessor32<float,2,torch::RestrictPtrTraits>()
//             );
//         }else if(val_dim==4){
//             slice_with_collisions_no_precomputation<3, 4><<<blocks, blockSize>>>(
//                 nr_positions, 
//                 positions.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
//                 sliced_values_hom_tensor.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
//                 splatting_indices_tensor.packed_accessor32<int,1,torch::RestrictPtrTraits>(),   
//                 splatting_weights_tensor.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
//                 should_precompute_tensors_for_backward,   
//                 *(m_hash_table->m_impl),
//                 m_hash_table->m_values_tensor.packed_accessor32<float,2,torch::RestrictPtrTraits>()
//             );
//         }else if(val_dim==8){
//             slice_with_collisions_no_precomputation<3, 8><<<blocks, blockSize>>>(
//                 nr_positions, 
//                 positions.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
//                 sliced_values_hom_tensor.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
//                 splatting_indices_tensor.packed_accessor32<int,1,torch::RestrictPtrTraits>(),   
//                 splatting_weights_tensor.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
//                 should_precompute_tensors_for_backward,   
//                 *(m_hash_table->m_impl),
//                 m_hash_table->m_values_tensor.packed_accessor32<float,2,torch::RestrictPtrTraits>()
//             );
//         }else if(val_dim==32){
//             slice_with_collisions_no_precomputation<3, 32><<<blocks, blockSize>>>(
//                 nr_positions, 
//                 positions.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
//                 sliced_values_hom_tensor.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
//                 splatting_indices_tensor.packed_accessor32<int,1,torch::RestrictPtrTraits>(),   
//                 splatting_weights_tensor.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
//                 should_precompute_tensors_for_backward,   
//                 *(m_hash_table->m_impl),
//                 m_hash_table->m_values_tensor.packed_accessor32<float,2,torch::RestrictPtrTraits>()
//             );
//         }else{
//             LOG(FATAL) <<"Instantiation for this val_dim doesnt exist. Please add it to the source code here.";
//         }
//     }else{
//         LOG(FATAL) <<"Instantiation for this pos_dim doesnt exist. Please add it to the source code here.";
//     }


   

//     // TIME_END("slice");

//     //transpose the values for returning them
//     // m_hash_table->m_values_tensor=m_hash_table->m_values_tensor.transpose(0,1).contiguous();;
//     // m_hash_table->update_impl();
//     // sliced_values_hom_tensor=sliced_values_hom_tensor.transpose(0,1).contiguous();;


//     auto ret = std::make_tuple (sliced_values_hom_tensor, splatting_indices_tensor, splatting_weights_tensor ); 
//     return ret;
// }

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> Lattice::slice_with_collisions_standalone_no_precomputation_fast(torch::Tensor& lattice_values, torch::Tensor& positions_raw, const bool should_precompute_tensors_for_backward){
    check_positions(positions_raw); 
    // CHECK(val_dim()>0) << "m_val_dim is 0 or lwoer. We have to splat something first so that we have values from where to slice. Val dim is " << val_dim();
    int nr_positions=positions_raw.size(0);
    int pos_dim=positions_raw.size(1);
    int val_dim=lattice_values.size(1);
    int lattice_capacity=lattice_values.size(0);
    CHECK(pos_dim==this->pos_dim()) << " The position dimension do not coreespond with the ones we used for creating the lattice";


     //to cuda
    positions_raw=positions_raw.to("cuda");
    m_sigmas_tensor=m_sigmas_tensor.to("cuda");

    Tensor positions=positions_raw; //the sigma scaling is done inside the kernel

    //initialize the output values 
    Tensor sliced_values_hom_tensor=torch::empty({nr_positions, val_dim }, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0) );

    //recalculate the splatting indices and weight for the backward pass of the slice
    Tensor splatting_indices_tensor;
    Tensor splatting_weights_tensor;
    if (should_precompute_tensors_for_backward){
        splatting_indices_tensor = torch::empty({ (pos_dim+1)* nr_positions }, torch::dtype(torch::kInt32).device(torch::kCUDA, 0) );
        splatting_weights_tensor = torch::empty({ (pos_dim+1)* nr_positions }, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0) );
        splatting_indices_tensor.fill_(-1);
        splatting_weights_tensor.fill_(-1);
    }else{
        splatting_indices_tensor = torch::empty({ 1 }, torch::dtype(torch::kInt32).device(torch::kCUDA, 0) );
        splatting_weights_tensor = torch::empty({ 1 }, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0) );
    }


    

    //make scalefactor
    // Eigen::VectorXd scale_factor_eigen;
    // scale_factor_eigen.resize(pos_dim);
    // double invStdDev = 1.0;
    // for (int i = 0; i < pos_dim; i++) {
    //     scale_factor_eigen(i) =  1.0 / (std::sqrt((double) (i + 1) * (i + 2))) * invStdDev;
    //     scale_factor_eigen(i)=scale_factor_eigen(i)/m_sigmas[i];
    // }
    // Tensor scale_factor_tensor=eigen2tensor(scale_factor_eigen.cast<float>()).cuda();
    // scale_factor_tensor=scale_factor_tensor.flatten();


    //do it directly in cuda
    // TODO it should probbly be done just once at the creation of the encoding
    Tensor scale_factor_tensor=torch::empty({pos_dim }, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0) );
    double invStdDev = 1.0;
    for (int i = 0; i < pos_dim; i++) {
        scale_factor_tensor[i] =  1.0 / (std::sqrt((double) (i + 1) * (i + 2))) * invStdDev;
        scale_factor_tensor[i]=scale_factor_tensor[i]/m_sigmas[i];
    }




    // bool use_transpose=true;
    // Tensor sliced_values_hom_tensor;
    //this makes the access to these more coalesced so the slicing itself is faster but this transpose and making into a contiguous array just makes everything slower in the end so we disable it
    //either way there is little difference in the slicing speed when the nr of features is low
    // if (use_transpose){
        // sliced_values_hom_tensor=torch::empty({ this->val_dim(), nr_positions }, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0) );
        // sliced_values_hom_tensor=sliced_values_hom_tensor.transpose(0,1);
    // }else{
        //  sliced_values_hom_tensor=torch::empty({nr_positions, this->val_dim() }, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0) );
    // }





    dim3 blocks((nr_positions - 1) / BLOCK_SIZE + 1, 1, 1);
    dim3 blockSize(BLOCK_SIZE, 1, 1);
    blockSize.y = 1;

    if (pos_dim==3){
        if(val_dim==2){
            slice_with_collisions_no_precomputation_fast<3, 2><<<blocks, blockSize>>>(
                nr_positions, 
                lattice_capacity,
                positions.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                lattice_values.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                scale_factor_tensor.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
                sliced_values_hom_tensor.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                splatting_indices_tensor.packed_accessor32<int,1,torch::RestrictPtrTraits>(),   
                splatting_weights_tensor.packed_accessor32<float,1,torch::RestrictPtrTraits>(),   
                should_precompute_tensors_for_backward
                // *(m_hash_table->m_impl),
                // m_hash_table->m_values_tensor.packed_accessor32<float,2,torch::RestrictPtrTraits>()
            );
        }else if(val_dim==4){
            slice_with_collisions_no_precomputation_fast<3, 4><<<blocks, blockSize>>>(
                nr_positions, 
                lattice_capacity,
                positions.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                lattice_values.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                scale_factor_tensor.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
                sliced_values_hom_tensor.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                splatting_indices_tensor.packed_accessor32<int,1,torch::RestrictPtrTraits>(),   
                splatting_weights_tensor.packed_accessor32<float,1,torch::RestrictPtrTraits>(),   
                should_precompute_tensors_for_backward
                // *(m_hash_table->m_impl),
                // m_hash_table->m_values_tensor.packed_accessor32<float,2,torch::RestrictPtrTraits>()
            );
        }else{
            LOG(FATAL) <<"Instantiation for this val_dim doesnt exist. Please add it to the source code here.";
        }
    }else{
        LOG(FATAL) <<"Instantiation for this pos_dim doesnt exist. Please add it to the source code here.";
    }


   


    auto ret = std::make_tuple (sliced_values_hom_tensor, splatting_indices_tensor, splatting_weights_tensor ); 
    return ret;
}

// std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> Lattice::slice_with_collisions_standalone_no_precomputation_fast_mr_loop(const std::vector<torch::Tensor>& lattice_values_list, const std::vector<float> sigmas_list, torch::Tensor& positions_raw, const bool should_precompute_tensors_for_backward){
//     check_positions(positions_raw); 
//     // CHECK(val_dim()>0) << "m_val_dim is 0 or lwoer. We have to splat something first so that we have values from where to slice. Val dim is " << val_dim();
//     int nr_positions=positions_raw.size(0);
//     int pos_dim=positions_raw.size(1);
//     CHECK(pos_dim==this->pos_dim()) << " The position dimension do not coreespond with the ones we used for creating the lattice";
//     //we assume that all the lattice values have the same shape
//     int nr_resolutions=lattice_values_list.size();
//     for(int i=0; i<nr_resolutions-1; i++){
//         CHECK( lattice_values_list[i].sizes()==lattice_values_list[i+1].sizes() ) << "We assume all lattice values match in shape over all the resolutions";
//     }
//     int lattice_capacity=lattice_values_list[0].size(0);
//     int val_dim=lattice_values_list[0].size(1);

//      //to cuda
//     positions_raw=positions_raw.to("cuda");
//     m_sigmas_tensor=m_sigmas_tensor.to("cuda");

//     Tensor positions=positions_raw; //the sigma scaling is done inside the kernel
    


//     //initialize the output values 
//     // Tensor sliced_values_hom_tensor=torch::empty({nr_positions, nr_resolutions* val_dim }, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0) );
//     Tensor sliced_values_hom_tensor=torch::empty({nr_resolutions, nr_positions, val_dim }, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0) );
//     // Tensor sliced_values_hom_tensor=torch::empty({nr_resolutions, val_dim, nr_positions }, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0) );

//     //recalculate the splatting indices and weight for the backward pass of the slice
//     Tensor splatting_indices_tensor;
//     Tensor splatting_weights_tensor;
//     if (should_precompute_tensors_for_backward){
//         splatting_indices_tensor = torch::empty({ nr_resolutions,  nr_positions, (pos_dim+1) }, torch::dtype(torch::kInt32).device(torch::kCUDA, 0) );
//         splatting_weights_tensor = torch::empty({ nr_resolutions,  nr_positions, (pos_dim+1) }, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0) );
//         splatting_indices_tensor.fill_(-1);
//         splatting_weights_tensor.fill_(-1);
//     }else{
//         splatting_indices_tensor = torch::empty({ 1,1,1 }, torch::dtype(torch::kInt32).device(torch::kCUDA, 0) );
//         splatting_weights_tensor = torch::empty({ 1,1,1 }, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0) );
//     }


//     bool use_transpose=false;
//     // Tensor sliced_values_hom_tensor;
//     // this makes the access to these more coalesced so the slicing itself is faster but this transpose and making into a contiguous array just makes everything slower in the end so we disable it
//     // either way there is little difference in the slicing speed when the nr of features is low
//     if (use_transpose){
//         // sliced_values_hom_tensor=torch::empty({ this->val_dim(), nr_positions }, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0) );
//         // sliced_values_hom_tensor=sliced_values_hom_tensor.transpose(0,1);
//         positions=positions.transpose(0,1).contiguous().transpose(0,1);
//         // positions=positions.transpose(0,1).contiguous();
//     }else{
//         //  sliced_values_hom_tensor=torch::empty({nr_positions, this->val_dim() }, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0) );
//     }
//     // VLOG(1) << "posituions strides" << positions.strides();


//     //SLICE each resolution
//     for(int res_idx=0; res_idx<nr_resolutions; res_idx++){

//         this->set_sigma(sigmas_list[res_idx]);

//         // //make scalefactor
//         // Eigen::VectorXd scale_factor_eigen;
//         // scale_factor_eigen.resize(pos_dim);
//         // double invStdDev = 1.0;
//         // for (int i = 0; i < pos_dim; i++) {
//         //     scale_factor_eigen(i) =  1.0 / (std::sqrt((double) (i + 1) * (i + 2))) * invStdDev;
//         //     scale_factor_eigen(i)=scale_factor_eigen(i)/m_sigmas[i];
//         // }
//         // Tensor scale_factor_tensor=eigen2tensor(scale_factor_eigen.cast<float>()).cuda();
//         // scale_factor_tensor=scale_factor_tensor.flatten();

//          //do it directly in cuda
//         // TODO it should probbly be done just once at the creation of the encoding
//         Tensor scale_factor_tensor=torch::empty({pos_dim }, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0) );
//         double invStdDev = 1.0;
//         for (int i = 0; i < pos_dim; i++) {
//             scale_factor_tensor[i] =  1.0 / (std::sqrt((double) (i + 1) * (i + 2))) * invStdDev;
//             scale_factor_tensor[i]=scale_factor_tensor[i]/m_sigmas[i];
//         }


//         dim3 blocks((nr_positions - 1) / BLOCK_SIZE + 1, 1, 1);
//         dim3 blockSize(BLOCK_SIZE, 1, 1);
//         blockSize.y = 1;

//         //get the chuk of the sliced values that we output into
//         // Tensor sliced_values_hom_tensor_3D=sliced_values_hom_tensor.view({nr_positions, nr_resolutions, val_dim});
//         // Tensor sliced_values_chunk=sliced_values_hom_tensor_3D.slice(1, res_idx, res_idx+1);
//         // sliced_values_chunk=sliced_values_chunk.view({nr_positions, val_dim});

//         //attempt 2
//         Tensor sliced_values_hom_tensor_3D=sliced_values_hom_tensor;
//         Tensor sliced_values_chunk=sliced_values_hom_tensor_3D.slice(0, res_idx, res_idx+1);
//         sliced_values_chunk=sliced_values_chunk.view({nr_positions, val_dim});
//         // sliced_values_chunk=sliced_values_chunk.view({val_dim, nr_positions});
//         // VLOG(1) << "3D shape" << sliced_values_hom_tensor_3D.sizes();
//         // VLOG(1) << "chunk " << sliced_values_chunk.sizes();

//         //get a chunck of the splatting indices and weights
//         Tensor splatting_indices_tensor_chunk=splatting_indices_tensor.slice(0, res_idx, res_idx+1);
//         splatting_indices_tensor_chunk=splatting_indices_tensor_chunk.flatten();
//         Tensor splatting_weights_tensor_chunk=splatting_weights_tensor.slice(0, res_idx, res_idx+1);
//         splatting_weights_tensor_chunk=splatting_weights_tensor_chunk.flatten();



//         Tensor lattice_values=lattice_values_list[res_idx];

        

//         if (pos_dim==3){
//             if(val_dim==2){
//                 slice_with_collisions_no_precomputation_fast_mr_loop<3, 2><<<blocks, blockSize>>>(
//                     nr_positions, 
//                     lattice_capacity,
//                     positions.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
//                     lattice_values.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
//                     scale_factor_tensor.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
//                     sliced_values_chunk.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
//                     splatting_indices_tensor_chunk.packed_accessor32<int,1,torch::RestrictPtrTraits>(),   
//                     splatting_weights_tensor_chunk.packed_accessor32<float,1,torch::RestrictPtrTraits>(),   
//                     should_precompute_tensors_for_backward
//                 );
//             }else if(val_dim==4){
//                 slice_with_collisions_no_precomputation_fast_mr_loop<3, 4><<<blocks, blockSize>>>(
//                     nr_positions, 
//                     lattice_capacity,
//                     positions.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
//                     lattice_values.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
//                     scale_factor_tensor.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
//                     sliced_values_chunk.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
//                     splatting_indices_tensor_chunk.packed_accessor32<int,1,torch::RestrictPtrTraits>(),   
//                     splatting_weights_tensor_chunk.packed_accessor32<float,1,torch::RestrictPtrTraits>(),   
//                     should_precompute_tensors_for_backward
//                 );
//             }else{
//                 LOG(FATAL) <<"Instantiation for this val_dim doesnt exist. Please add it to the source code here.";
//             }
//         }else{
//             LOG(FATAL) <<"Instantiation for this pos_dim doesnt exist. Please add it to the source code here.";
//         }
            


//     }

   






//     auto ret = std::make_tuple (sliced_values_hom_tensor, splatting_indices_tensor, splatting_weights_tensor ); 
//     return ret;

// }


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> Lattice::slice_with_collisions_standalone_no_precomputation_fast_mr_monolithic(const torch::Tensor& lattice_values_monolithic, const torch::Tensor& scale_factor, torch::Tensor& positions_raw, torch::Tensor& random_shift_monolithic, torch::Tensor& anneal_window, const bool concat_points, const float points_scaling, const bool require_lattice_values_grad, const bool require_positions_grad){

    // TIME_START("slice_prep");
    check_positions(positions_raw); 
    int nr_positions=positions_raw.size(0);
    int pos_dim=positions_raw.size(1);
    CHECK(pos_dim==this->pos_dim()) << " The position dimension do not coreespond with the ones we used for creating the lattice";
    //we assume that all the lattice values have the same shape
    // int nr_resolutions=sigmas_list.size();
    int nr_resolutions=lattice_values_monolithic.size(0);
    int lattice_capacity=lattice_values_monolithic.size(1);
    int val_dim=lattice_values_monolithic.size(2);
    CHECK(random_shift_monolithic.size(0)==nr_resolutions ) <<"Random shift should have the first dimension the same as the nr of resolutions";
    CHECK(random_shift_monolithic.size(1)==pos_dim ) <<"Random shift should have the second dimension the same as the pos dim";
    //check the anneal window
    CHECK(anneal_window.size(0)==nr_resolutions ) <<"anneal_window should have the first dimension the same as the nr of resolutions";

     //to cuda
    positions_raw=positions_raw.to("cuda");
    m_sigmas_tensor=m_sigmas_tensor.to("cuda");
    random_shift_monolithic=random_shift_monolithic.to("cuda");
    anneal_window=anneal_window.to("cuda");

    Tensor positions=positions_raw; //the sigma scaling is done inside the kernel
    

    //if we concat also the points, we add a series of extra resolutions to contain those points
    int nr_resolutions_extra=0;
    if (concat_points){
        nr_resolutions_extra=std::ceil(float(pos_dim)/val_dim);
    }
    // VLOG(1) << "nr_resolutions_extra" <<nr_resolutions_extra;
    // TIME_END("slice_prep");



    //initialize the output values 
    // Tensor sliced_values_hom_tensor=torch::empty({nr_positions, nr_resolutions, val_dim }, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0) );
    // Tensor sliced_values_hom_tensor=torch::empty({nr_resolutions, nr_positions, val_dim }, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0) ); //fastest
    // TIME_START("slice_create_output");
    // #if LATTICE_HALF_PRECISION
        // Tensor sliced_values_hom_tensor=torch::empty({nr_resolutions+nr_resolutions_extra, val_dim, nr_positions }, torch::dtype(torch::kFloat16).device(torch::kCUDA, 0) );
    // #else
        Tensor sliced_values_hom_tensor=torch::empty({nr_resolutions+nr_resolutions_extra, val_dim, nr_positions }, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0) );
    // #endif
    // TIME_END("slice_create_output");

    //recalculate the splatting indices and weight for the backward pass of the slice
    // TIME_START("slice_create_indx_and_w");
    Tensor splatting_indices_tensor;
    Tensor splatting_weights_tensor;
    if (require_lattice_values_grad || require_positions_grad){
        splatting_indices_tensor = torch::empty({ nr_resolutions, nr_positions, (pos_dim+1) }, torch::dtype(torch::kInt32).device(torch::kCUDA, 0) );
        #if LATTICE_HALF_PRECISION
            splatting_weights_tensor = torch::empty({ nr_resolutions, nr_positions, (pos_dim+1) }, torch::dtype(torch::kFloat16).device(torch::kCUDA, 0) );
        #else 
            splatting_weights_tensor = torch::empty({ nr_resolutions, nr_positions, (pos_dim+1) }, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0) );
        #endif
        splatting_indices_tensor.fill_(-1);
        // splatting_weights_tensor.fill_(0);
    }else{
        splatting_indices_tensor = torch::empty({ 1,1,1 }, torch::dtype(torch::kInt32).device(torch::kCUDA, 0) );
        #if LATTICE_HALF_PRECISION
            splatting_weights_tensor = torch::empty({ 1,1,1 }, torch::dtype(torch::kFloat16).device(torch::kCUDA, 0) );
        #else
            splatting_weights_tensor = torch::empty({ 1,1,1 }, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0) );
        #endif
    }
    // TIME_END("slice_create_indx_and_w");



    //make scalefactor for each lvl
    // TIME_START("slice_create_scale_factor");
    // Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> scale_factor_eigen;
    // scale_factor_eigen.resize(nr_resolutions, pos_dim);
    // double invStdDev = 1.0;
    // for(int res_idx=0; res_idx<nr_resolutions; res_idx++){
    //     for (int i = 0; i < pos_dim; i++) {
    //         scale_factor_eigen(res_idx,i) =  1.0 / (std::sqrt((double) (i + 1) * (i + 2))) * invStdDev;
    //         scale_factor_eigen(res_idx,i)=scale_factor_eigen(res_idx,i)/ sigmas_list[res_idx];
    //         // VLOG(1) << "scalinbg by " << sigmas_list[res_idx];
    //     }
    // }
    // VLOG(1) << "scale_factor_eigen" << scale_factor_eigen;
    // Tensor scale_factor_tensor=Lattice::compute_scale_factor_tensor(sigmas_list, pos_dim);
    // scale_factor_tensor=scale_factor_tensor.view({nr_resolutions, pos_dim}); //nr_resolutuons x pos_dim
    Tensor scale_factor_tensor=scale_factor;
    // TIME_END("slice_create_scale_factor");
    // VLOG(1) << "scale_factor_tensor" << scale_factor_tensor;
    // VLOG(1) << "scale_factor tensor is " << scale_factor_tensor.sizes();

    //try constant memory
    // Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> scale_factor_eigen_f;
    // scale_factor_eigen_f=scale_factor_eigen.cast<float>();
    // cudaMemcpyToSymbol(scalings_constants, scale_factor_eigen_f.data(), sizeof(float)*pos_dim*nr_resolutions  );



    //try again with a monolithic kernel
    const dim3 blocks = { (unsigned int)div_round_up(nr_positions, BLOCK_SIZE), (unsigned int)(nr_resolutions+nr_resolutions_extra), 1 }; //the blocks are executed in order, first the blocks for the first resolution, then the second and so on
    

    //debug by wirting also the elevated ones
    // Tensor elevated=torch::empty({nr_resolutions, nr_positions, pos_dim+1 }, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0) );
    // Tensor elevated=torch::empty({nr_resolutions,  pos_dim+1, nr_positions }, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0) );

    //try to transpose some of the stuff
    // positions=positions.transpose(0,1).contiguous().transpose(0,1);

    // TIME_START("slice_forwarD_cuda");
    if (pos_dim==2){
        if(val_dim==2){
            slice_with_collisions_no_precomputation_fast_mr_monolithic<2, 2><<<blocks, BLOCK_SIZE>>>(
                nr_positions, 
                lattice_capacity,
                nr_resolutions,
                nr_resolutions_extra,
                positions.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                #if LATTICE_HALF_PRECISION
                    lattice_values_monolithic.packed_accessor32<at::Half,3,torch::RestrictPtrTraits>(),
                #else 
                    lattice_values_monolithic.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
                #endif
                scale_factor_tensor.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                random_shift_monolithic.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                anneal_window.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
                // #if LATTICE_HALF_PRECISION
                    // sliced_values_hom_tensor.packed_accessor32<at::Half,3,torch::RestrictPtrTraits>(),
                // #else
                    sliced_values_hom_tensor.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
                // #endif
                // elevated.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
                splatting_indices_tensor.packed_accessor32<int,3,torch::RestrictPtrTraits>(),   
                #if LATTICE_HALF_PRECISION
                    splatting_weights_tensor.packed_accessor32<at::Half,3,torch::RestrictPtrTraits>(),   
                #else 
                    splatting_weights_tensor.packed_accessor32<float,3,torch::RestrictPtrTraits>(),   
                #endif
                concat_points,
                points_scaling,
                require_lattice_values_grad,
                require_positions_grad
            );
        }else{
            LOG(FATAL) <<"Instantiation for this val_dim doesnt exist. Please add it to the source code here.";
        }
    }else if (pos_dim==3){
        // LOG(FATAL) <<"Instantiation for this pos_dim doesnt exist. Please add it to the source code here.";
        if(val_dim==2){
            slice_with_collisions_no_precomputation_fast_mr_monolithic<3, 2><<<blocks, BLOCK_SIZE>>>(
                nr_positions, 
                lattice_capacity,
                nr_resolutions,
                nr_resolutions_extra,
                positions.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                // lattice_values_monolithic.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
                #if LATTICE_HALF_PRECISION
                    lattice_values_monolithic.packed_accessor32<at::Half,3,torch::RestrictPtrTraits>(),
                #else 
                    lattice_values_monolithic.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
                #endif
                scale_factor_tensor.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                random_shift_monolithic.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                anneal_window.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
                // sliced_values_hom_tensor.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
                // #if LATTICE_HALF_PRECISION
                    // sliced_values_hom_tensor.packed_accessor32<at::Half,3,torch::RestrictPtrTraits>(),
                // #else
                    sliced_values_hom_tensor.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
                // #endif
                // elevated.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
                splatting_indices_tensor.packed_accessor32<int,3,torch::RestrictPtrTraits>(),   
                #if LATTICE_HALF_PRECISION
                    splatting_weights_tensor.packed_accessor32<at::Half,3,torch::RestrictPtrTraits>(),   
                #else 
                    splatting_weights_tensor.packed_accessor32<float,3,torch::RestrictPtrTraits>(),   
                #endif
                concat_points,
                points_scaling,
                require_lattice_values_grad,
                require_positions_grad
            );
        }else{
            LOG(FATAL) <<"Instantiation for this val_dim doesnt exist. Please add it to the source code here.";
        }
    }else if (pos_dim==4){
        // LOG(FATAL) <<"Instantiation for this pos_dim doesnt exist. Please add it to the source code here.";
        if(val_dim==2){
            slice_with_collisions_no_precomputation_fast_mr_monolithic<4, 2><<<blocks, BLOCK_SIZE>>>(
                nr_positions, 
                lattice_capacity,
                nr_resolutions,
                nr_resolutions_extra,
                positions.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                // lattice_values_monolithic.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
                #if LATTICE_HALF_PRECISION
                    lattice_values_monolithic.packed_accessor32<at::Half,3,torch::RestrictPtrTraits>(),
                #else 
                    lattice_values_monolithic.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
                #endif
                scale_factor_tensor.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                random_shift_monolithic.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                anneal_window.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
                // sliced_values_hom_tensor.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
                // #if LATTICE_HALF_PRECISION
                    // sliced_values_hom_tensor.packed_accessor32<at::Half,3,torch::RestrictPtrTraits>(),
                // #else
                    sliced_values_hom_tensor.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
                // #endif
                // elevated.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
                splatting_indices_tensor.packed_accessor32<int,3,torch::RestrictPtrTraits>(),   
                #if LATTICE_HALF_PRECISION
                    splatting_weights_tensor.packed_accessor32<at::Half,3,torch::RestrictPtrTraits>(),   
                #else 
                    splatting_weights_tensor.packed_accessor32<float,3,torch::RestrictPtrTraits>(),   
                #endif
                concat_points,
                points_scaling,
                require_lattice_values_grad,
                require_positions_grad
            );
        }else{
            LOG(FATAL) <<"Instantiation for this val_dim doesnt exist. Please add it to the source code here.";
        }
    }else if (pos_dim==5){
        // LOG(FATAL) <<"Instantiation for this pos_dim doesnt exist. Please add it to the source code here.";
        if(val_dim==2){
            slice_with_collisions_no_precomputation_fast_mr_monolithic<5, 2><<<blocks, BLOCK_SIZE>>>(
                nr_positions, 
                lattice_capacity,
                nr_resolutions,
                nr_resolutions_extra,
                positions.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                // lattice_values_monolithic.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
                #if LATTICE_HALF_PRECISION
                    lattice_values_monolithic.packed_accessor32<at::Half,3,torch::RestrictPtrTraits>(),
                #else 
                    lattice_values_monolithic.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
                #endif
                scale_factor_tensor.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                random_shift_monolithic.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                anneal_window.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
                // sliced_values_hom_tensor.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
                // #if LATTICE_HALF_PRECISION
                    // sliced_values_hom_tensor.packed_accessor32<at::Half,3,torch::RestrictPtrTraits>(),
                // #else
                    sliced_values_hom_tensor.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
                // #endif
                // elevated.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
                splatting_indices_tensor.packed_accessor32<int,3,torch::RestrictPtrTraits>(),   
                #if LATTICE_HALF_PRECISION
                    splatting_weights_tensor.packed_accessor32<at::Half,3,torch::RestrictPtrTraits>(),   
                #else 
                    splatting_weights_tensor.packed_accessor32<float,3,torch::RestrictPtrTraits>(),   
                #endif
                concat_points,
                points_scaling,
                require_lattice_values_grad,
                require_positions_grad
            );
        }else{
            LOG(FATAL) <<"Instantiation for this val_dim doesnt exist. Please add it to the source code here.";
        }
    }else if (pos_dim==6){
        // LOG(FATAL) <<"Instantiation for this pos_dim doesnt exist. Please add it to the source code here.";
        if(val_dim==2){
            slice_with_collisions_no_precomputation_fast_mr_monolithic<6, 2><<<blocks, BLOCK_SIZE>>>(
                nr_positions, 
                lattice_capacity,
                nr_resolutions,
                nr_resolutions_extra,
                positions.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                // lattice_values_monolithic.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
                #if LATTICE_HALF_PRECISION
                    lattice_values_monolithic.packed_accessor32<at::Half,3,torch::RestrictPtrTraits>(),
                #else 
                    lattice_values_monolithic.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
                #endif
                scale_factor_tensor.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                random_shift_monolithic.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                anneal_window.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
                // sliced_values_hom_tensor.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
                // #if LATTICE_HALF_PRECISION
                    // sliced_values_hom_tensor.packed_accessor32<at::Half,3,torch::RestrictPtrTraits>(),
                // #else
                    sliced_values_hom_tensor.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
                // #endif
                // elevated.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
                splatting_indices_tensor.packed_accessor32<int,3,torch::RestrictPtrTraits>(),   
                #if LATTICE_HALF_PRECISION
                    splatting_weights_tensor.packed_accessor32<at::Half,3,torch::RestrictPtrTraits>(),   
                #else 
                    splatting_weights_tensor.packed_accessor32<float,3,torch::RestrictPtrTraits>(),   
                #endif
                concat_points,
                points_scaling,
                require_lattice_values_grad,
                require_positions_grad
            );
        }else{
            LOG(FATAL) <<"Instantiation for this val_dim doesnt exist. Please add it to the source code here.";
        }
    }else{
        LOG(FATAL) <<"Instantiation for this pos_dim doesnt exist. Please add it to the source code here.";
    }
    // TIME_END("slice_forwarD_cuda");









    // //SLICE each resolution
    // for(int res_idx=0; res_idx<nr_resolutions; res_idx++){

    //     this->set_sigma(sigmas_list[res_idx]);

    //     //make scalefactor
    //     Eigen::VectorXd scale_factor_eigen;
    //     scale_factor_eigen.resize(pos_dim);
    //     double invStdDev = 1.0;
    //     for (int i = 0; i < pos_dim; i++) {
    //         scale_factor_eigen(i) =  1.0 / (std::sqrt((double) (i + 1) * (i + 2))) * invStdDev;
    //         scale_factor_eigen(i)=scale_factor_eigen(i)/m_sigmas[i];
    //     }
    //     Tensor scale_factor_tensor=eigen2tensor(scale_factor_eigen.cast<float>()).cuda();
    //     scale_factor_tensor=scale_factor_tensor.flatten();


    //     dim3 blocks((nr_positions - 1) / BLOCK_SIZE + 1, 1, 1);
    //     dim3 blockSize(BLOCK_SIZE, 1, 1);
    //     blockSize.y = 1;

    //     //get the chuk of the sliced values that we output into
    //     // Tensor sliced_values_hom_tensor_3D=sliced_values_hom_tensor.view({nr_positions, nr_resolutions, val_dim});
    //     // Tensor sliced_values_chunk=sliced_values_hom_tensor_3D.slice(1, res_idx, res_idx+1);
    //     // sliced_values_chunk=sliced_values_chunk.view({nr_positions, val_dim});

    //     //attempt 2
    //     Tensor sliced_values_hom_tensor_3D=sliced_values_hom_tensor;
    //     Tensor sliced_values_chunk=sliced_values_hom_tensor_3D.slice(0, res_idx, res_idx+1);
    //     sliced_values_chunk=sliced_values_chunk.view({nr_positions, val_dim});
    //     // sliced_values_chunk=sliced_values_chunk.view({val_dim, nr_positions});
    //     // VLOG(1) << "3D shape" << sliced_values_hom_tensor_3D.sizes();
    //     // VLOG(1) << "chunk " << sliced_values_chunk.sizes();

    //     //get a chunck of the splatting indices and weights
    //     Tensor splatting_indices_tensor_chunk=splatting_indices_tensor.slice(0, res_idx, res_idx+1);
    //     splatting_indices_tensor_chunk=splatting_indices_tensor_chunk.flatten();
    //     Tensor splatting_weights_tensor_chunk=splatting_weights_tensor.slice(0, res_idx, res_idx+1);
    //     splatting_weights_tensor_chunk=splatting_weights_tensor_chunk.flatten();



    //     Tensor lattice_values=lattice_values_list[res_idx];

        

    //     if (pos_dim==3){
    //         if(val_dim==4){
    //             slice_with_collisions_no_precomputation_fast_mr_monolithic<3, 4><<<blocks, blockSize>>>(
    //                 nr_positions, 
    //                 lattice_capacity,
    //                 positions.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
    //                 lattice_values.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
    //                 scale_factor_tensor.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
    //                 sliced_values_chunk.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
    //                 splatting_indices_tensor_chunk.packed_accessor32<int,1,torch::RestrictPtrTraits>(),   
    //                 splatting_weights_tensor_chunk.packed_accessor32<float,1,torch::RestrictPtrTraits>(),   
    //                 should_precompute_tensors_for_backward
    //             );
    //         }else{
    //             LOG(FATAL) <<"Instantiation for this val_dim doesnt exist. Please add it to the source code here.";
    //         }
    //     }else{
    //         LOG(FATAL) <<"Instantiation for this pos_dim doesnt exist. Please add it to the source code here.";
    //     }
            


    // }

    // TIME_START("switch_to_float");
    if (Lattice::is_half_precision()){ //return the values as flaot32 because we want the gradient to also be float32
        // sliced_values_hom_tensor=sliced_values_hom_tensor.to(torch::kFloat32);
    }
    // TIME_END("switch_to_float");





    auto ret = std::make_tuple (sliced_values_hom_tensor, splatting_indices_tensor, splatting_weights_tensor ); 
    return ret;

}




//backward
// void Lattice::slice_backwards_standalone_with_precomputation_no_homogeneous(torch::Tensor& positions_raw, const torch::Tensor& grad_sliced_values, torch::Tensor& splatting_indices_tensor, torch::Tensor& splatting_weights_tensor){

//     check_positions(positions_raw); 
//     CHECK(val_dim()>0) << "m_val_dim is 0 or lwoer. We have to splat something first so that we have values from where to slice. Val dim is " << val_dim();
//     int nr_positions=positions_raw.size(0);
//     int pos_dim=positions_raw.size(1);
//     CHECK(pos_dim==this->pos_dim()) << " The position dimension do not coreespond with the ones we used for creating the lattice";
//     CHECK(grad_sliced_values.dim()==2) <<"grad_sliced_values should be nr_positions x m_val_dim, so it should have 2 dimensions. However it has "<< grad_sliced_values.dim();
//     CHECK(grad_sliced_values.is_contiguous()) << "Grad sliced values needs to be contiguous. Please call .contiguous() on it";
//     splatting_indices_tensor=splatting_indices_tensor.contiguous();
//     splatting_weights_tensor=splatting_weights_tensor.contiguous();

//     m_hash_table->m_values_tensor=torch::zeros({nr_lattice_vertices(), grad_sliced_values.size(1)},  torch::dtype(torch::kFloat32).device(torch::kCUDA, 0)  );
//     m_hash_table->update_impl();




//     //transpose the values
//     int val_dim=m_hash_table->m_values_tensor.size(1);



//     TIME_START("slice_back");
//     // m_impl->slice_backwards_standalone_with_precomputation_no_homogeneous(grad_sliced_values.data_ptr<float>(), splatting_indices_tensor.data_ptr<int>(), splatting_weights_tensor.data_ptr<float>(), this->pos_dim(), this->val_dim(), nr_positions, *(m_hash_table->m_impl) );

//     dim3 blocks((nr_positions - 1) / BLOCK_SIZE + 1, 1, 1);
//     dim3 blockSize(BLOCK_SIZE, 1, 1);
//     blockSize.y = 1;
//     slice_backwards_with_precomputation_no_homogeneous<3><<<blocks, blockSize>>>(
//         nr_positions, 
//         val_dim, 
//         grad_sliced_values.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
//         splatting_indices_tensor.packed_accessor32<int,1,torch::RestrictPtrTraits>(),   
//         splatting_weights_tensor.packed_accessor32<float,1,torch::RestrictPtrTraits>(),   
//         m_hash_table->m_values_tensor.packed_accessor32<float,2,torch::RestrictPtrTraits>()
//     );


//     // const int nr_positions,
//     // const int val_dim, 
//     // const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> grad_sliced_values,
//     // const torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> splatting_indices,
//     // const torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> splatting_weights,
//     // torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> val_out


//     TIME_END("slice_back");



//     //transpose back

// }

torch::Tensor Lattice::slice_backwards_standalone_with_precomputation_no_homogeneous_mr_monolithic(torch::Tensor& positions_raw, const torch::Tensor& grad_sliced_values_monolithic, torch::Tensor& splatting_indices_tensor, torch::Tensor& splatting_weights_tensor){

    check_positions(positions_raw); 
    // CHECK(val_dim()>0) << "m_val_dim is 0 or lwoer. We have to splat something first so that we have values from where to slice. Val dim is " << val_dim();
    int nr_positions=positions_raw.size(0);
    int pos_dim=positions_raw.size(1);
    CHECK(pos_dim==this->pos_dim()) << " The position dimension do not coreespond with the ones we used for creating the lattice";
    CHECK(grad_sliced_values_monolithic.dim()==3) <<"grad_sliced_values_monolithic should be nr_resolutions x val_dim x nr_positions, so it should have 3 dimensions. However it has "<< grad_sliced_values_monolithic.dim();
    // CHECK(grad_sliced_values_monolithic.is_contiguous()) << "Grad sliced values needs to be contiguous. Please call .contiguous() on it";
    // splatting_indices_tensor=splatting_indices_tensor.contiguous();
    // splatting_weights_tensor=splatting_weights_tensor.contiguous();
    int nr_resolutions=grad_sliced_values_monolithic.size(0);
    int val_dim=grad_sliced_values_monolithic.size(1);
    CHECK(nr_positions==grad_sliced_values_monolithic.size(2)) << "The nr of positions should match between the input positions and the sliced values";

    

    // nr_resolutions x nr_lattice_vertices x nr_lattice_featues
    // Tensor lattice_values_monolithic=torch::zeros({ nr_resolutions, nr_lattice_vertices(), val_dim  },  torch::dtype(torch::kFloat32).device(torch::kCUDA, 0)  );
    // TIME_START("create_backgrad");
    #if LATTICE_HALF_PRECISION
        // Tensor lattice_values_monolithic=torch::zeros({ nr_resolutions, val_dim, m_capacity },  torch::dtype(torch::kFloat16).device(torch::kCUDA, 0)  );
        Tensor lattice_values_monolithic=torch::zeros({ nr_resolutions, m_capacity, val_dim },  torch::dtype(torch::kFloat16).device(torch::kCUDA, 0)  );
    #else
        // Tensor lattice_values_monolithic=torch::zeros({ nr_resolutions, val_dim, m_capacity },  torch::dtype(torch::kFloat32).device(torch::kCUDA, 0)  );
        Tensor lattice_values_monolithic=torch::zeros({ nr_resolutions, m_capacity, val_dim },  torch::dtype(torch::kFloat32).device(torch::kCUDA, 0)  );
    #endif
    // TIME_END("create_backgrad");

    //test some tranposes
    // grad_sliced_values_monolithic=grad_sliced_values_monolithic.transpose(1,2).contiguous().transpose(1,2);


    //try to permute the positiosn so that when we accumulate back we have little chance of two adyacent threads to do the same atomicadd on the same lattice vertex
    // torch::Tensor shuffled_indices = torch::randperm(nr_positions, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA, 0));


    //attempt1 do each lvl individually
    const dim3 blocks = { (unsigned int)div_round_up(nr_positions, BLOCK_SIZE_BACK), (unsigned int)nr_resolutions, 1 }; //the blocks are executed in order, first the blocks for the first resolution, then the second and so on
    // TIME_START("slice_back_cuda");
    if (pos_dim==2){
        if(val_dim==1){
            // slice_backwards_with_precomputation_no_homogeneous_mr_monolithic<3,1><<<blocks, BLOCK_SIZE_BACK>>>(
            //     nr_positions, 
            //     grad_sliced_values_monolithic.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
            //     splatting_indices_tensor.packed_accessor32<int,3,torch::RestrictPtrTraits>(),   
            //     splatting_weights_tensor.packed_accessor32<at::Half,3,torch::RestrictPtrTraits>(),   
            //     lattice_values_monolithic.packed_accessor32<float,3,torch::RestrictPtrTraits>()
            // );
            LOG(FATAL) <<"I'll implement it later";
        }else if(val_dim==2){
            slice_backwards_with_precomputation_no_homogeneous_mr_monolithic<2,2><<<blocks, BLOCK_SIZE_BACK>>>(
                nr_positions, 
                // grad_sliced_values_monolithic.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
                #if LATTICE_HALF_PRECISION
                    grad_sliced_values_monolithic.packed_accessor32<at::Half,3,torch::RestrictPtrTraits>(),
                #else
                    grad_sliced_values_monolithic.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
                #endif
                splatting_indices_tensor.packed_accessor32<int,3,torch::RestrictPtrTraits>(),   
                // splatting_weights_tensor.packed_accessor32<float,3,torch::RestrictPtrTraits>(),   
                #if LATTICE_HALF_PRECISION
                    splatting_weights_tensor.packed_accessor32<at::Half,3,torch::RestrictPtrTraits>(),   
                #else 
                    splatting_weights_tensor.packed_accessor32<float,3,torch::RestrictPtrTraits>(),   
                #endif
                #if LATTICE_HALF_PRECISION
                    lattice_values_monolithic.packed_accessor32<at::Half,3,torch::RestrictPtrTraits>()
                #else   
                    lattice_values_monolithic.packed_accessor32<float,3,torch::RestrictPtrTraits>()
                #endif
            );
        }else if(val_dim==4){
            // slice_backwards_with_precomputation_no_homogeneous_mr_monolithic<3,4><<<blocks, BLOCK_SIZE_BACK>>>(
            //     nr_positions, 
            //     grad_sliced_values_monolithic.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
            //     splatting_indices_tensor.packed_accessor32<int,3,torch::RestrictPtrTraits>(),   
            //     splatting_weights_tensor.packed_accessor32<at::Half,3,torch::RestrictPtrTraits>(),   
            //     lattice_values_monolithic.packed_accessor32<float,3,torch::RestrictPtrTraits>()
            // );
            LOG(FATAL) <<"I'll implement it later";
        }else{
            LOG(FATAL) <<"Instantiation for this val_dim doesnt exist. Please add it to the source code here.";
        }
    }else if(pos_dim==3){
        // LOG(FATAL) <<"Instantiation for this pos_dim doesnt exist. Please add it to the source code here.";
        if(val_dim==2){
            slice_backwards_with_precomputation_no_homogeneous_mr_monolithic<3,2><<<blocks, BLOCK_SIZE_BACK>>>(
                nr_positions, 
                #if LATTICE_HALF_PRECISION
                    grad_sliced_values_monolithic.packed_accessor32<at::Half,3,torch::RestrictPtrTraits>(),
                #else
                    grad_sliced_values_monolithic.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
                #endif
                splatting_indices_tensor.packed_accessor32<int,3,torch::RestrictPtrTraits>(),   
                // splatting_weights_tensor.packed_accessor32<float,3,torch::RestrictPtrTraits>(),   
                #if LATTICE_HALF_PRECISION
                    splatting_weights_tensor.packed_accessor32<at::Half,3,torch::RestrictPtrTraits>(),   
                #else 
                    splatting_weights_tensor.packed_accessor32<float,3,torch::RestrictPtrTraits>(),   
                #endif
                #if LATTICE_HALF_PRECISION
                    lattice_values_monolithic.packed_accessor32<at::Half,3,torch::RestrictPtrTraits>()
                #else   
                    lattice_values_monolithic.packed_accessor32<float,3,torch::RestrictPtrTraits>()
                #endif
            );
        }else{
            LOG(FATAL) <<"Instantiation for this val_dim doesnt exist. Please add it to the source code here.";
        }
    }else if(pos_dim==4){
        // LOG(FATAL) <<"Instantiation for this pos_dim doesnt exist. Please add it to the source code here.";
        if(val_dim==2){
            slice_backwards_with_precomputation_no_homogeneous_mr_monolithic<4,2><<<blocks, BLOCK_SIZE_BACK>>>(
                nr_positions, 
                #if LATTICE_HALF_PRECISION
                    grad_sliced_values_monolithic.packed_accessor32<at::Half,3,torch::RestrictPtrTraits>(),
                #else
                    grad_sliced_values_monolithic.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
                #endif
                splatting_indices_tensor.packed_accessor32<int,3,torch::RestrictPtrTraits>(),   
                // splatting_weights_tensor.packed_accessor32<float,3,torch::RestrictPtrTraits>(),   
                #if LATTICE_HALF_PRECISION
                    splatting_weights_tensor.packed_accessor32<at::Half,3,torch::RestrictPtrTraits>(),   
                #else 
                    splatting_weights_tensor.packed_accessor32<float,3,torch::RestrictPtrTraits>(),   
                #endif
                #if LATTICE_HALF_PRECISION
                    lattice_values_monolithic.packed_accessor32<at::Half,3,torch::RestrictPtrTraits>()
                #else   
                    lattice_values_monolithic.packed_accessor32<float,3,torch::RestrictPtrTraits>()
                #endif
            );
        }else{
            LOG(FATAL) <<"Instantiation for this val_dim doesnt exist. Please add it to the source code here.";
        }
    }else if(pos_dim==5){
        // LOG(FATAL) <<"Instantiation for this pos_dim doesnt exist. Please add it to the source code here.";
        if(val_dim==2){
            slice_backwards_with_precomputation_no_homogeneous_mr_monolithic<5,2><<<blocks, BLOCK_SIZE_BACK>>>(
                nr_positions, 
                #if LATTICE_HALF_PRECISION
                    grad_sliced_values_monolithic.packed_accessor32<at::Half,3,torch::RestrictPtrTraits>(),
                #else
                    grad_sliced_values_monolithic.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
                #endif
                splatting_indices_tensor.packed_accessor32<int,3,torch::RestrictPtrTraits>(),   
                // splatting_weights_tensor.packed_accessor32<float,3,torch::RestrictPtrTraits>(),   
                #if LATTICE_HALF_PRECISION
                    splatting_weights_tensor.packed_accessor32<at::Half,3,torch::RestrictPtrTraits>(),   
                #else 
                    splatting_weights_tensor.packed_accessor32<float,3,torch::RestrictPtrTraits>(),   
                #endif
                #if LATTICE_HALF_PRECISION
                    lattice_values_monolithic.packed_accessor32<at::Half,3,torch::RestrictPtrTraits>()
                #else   
                    lattice_values_monolithic.packed_accessor32<float,3,torch::RestrictPtrTraits>()
                #endif
            );
        }else{
            LOG(FATAL) <<"Instantiation for this val_dim doesnt exist. Please add it to the source code here.";
        }
    }else if(pos_dim==6){
        // LOG(FATAL) <<"Instantiation for this pos_dim doesnt exist. Please add it to the source code here.";
        if(val_dim==2){
            slice_backwards_with_precomputation_no_homogeneous_mr_monolithic<6,2><<<blocks, BLOCK_SIZE_BACK>>>(
                nr_positions, 
                #if LATTICE_HALF_PRECISION
                    grad_sliced_values_monolithic.packed_accessor32<at::Half,3,torch::RestrictPtrTraits>(),
                #else
                    grad_sliced_values_monolithic.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
                #endif
                splatting_indices_tensor.packed_accessor32<int,3,torch::RestrictPtrTraits>(),   
                // splatting_weights_tensor.packed_accessor32<float,3,torch::RestrictPtrTraits>(),   
                #if LATTICE_HALF_PRECISION
                    splatting_weights_tensor.packed_accessor32<at::Half,3,torch::RestrictPtrTraits>(),   
                #else 
                    splatting_weights_tensor.packed_accessor32<float,3,torch::RestrictPtrTraits>(),   
                #endif
                #if LATTICE_HALF_PRECISION
                    lattice_values_monolithic.packed_accessor32<at::Half,3,torch::RestrictPtrTraits>()
                #else   
                    lattice_values_monolithic.packed_accessor32<float,3,torch::RestrictPtrTraits>()
                #endif
            );
        }else{
            LOG(FATAL) <<"Instantiation for this val_dim doesnt exist. Please add it to the source code here.";
        }
    }else{
        LOG(FATAL) <<"Instantiation for this pos_dim doesnt exist. Please add it to the source code here.";
    }
    // TIME_END("slice_back_cuda");


    //attempt 3, the thread does all the values and all the levels for that positions. It's easier with this to accumulate the gradient into the positions because it can be done with one thread attenting to the positions over all values and all levels
    //it's for nwo way slower
    // const dim3 blocks = { div_round_up(nr_positions, BLOCK_SIZE), 1, 1 }; //the blocks are executed in order, first the blocks for the first resolution, then the second and so on
    // if (pos_dim==3){
    //     if(val_dim==2){
    //         slice_backwards_with_precomputation_no_homogeneous_mr_monolithic_full_pos<3,2><<<blocks, BLOCK_SIZE>>>(
    //             nr_positions, 
    //             nr_resolutions,
    //             grad_sliced_values_monolithic.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
    //             splatting_indices_tensor.packed_accessor32<int,3,torch::RestrictPtrTraits>(),   
    //             splatting_weights_tensor.packed_accessor32<float,3,torch::RestrictPtrTraits>(),   
    //             lattice_values_monolithic.packed_accessor32<float,3,torch::RestrictPtrTraits>()
    //         );
    //     }else{
    //         LOG(FATAL) <<"Instantiation for this val_dim doesnt exist. Please add it to the source code here.";
    //     }
    // }else{
    //     LOG(FATAL) <<"Instantiation for this pos_dim doesnt exist. Please add it to the source code here.";
    // }


   
    // //attempt 2 with shared memory
    // const dim3 blocks = { div_round_up(nr_positions, BLOCK_SIZE), pos_dim+1, nr_resolutions };
    // if (pos_dim==3){
    //     if(val_dim==2){
    //         slice_backwards_with_precomputation_no_homogeneous_mr_monolithic_sharedmem<3,2><<<blocks, BLOCK_SIZE>>>(
    //             nr_positions, 
    //             grad_sliced_values_monolithic.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
    //             splatting_indices_tensor.packed_accessor32<int,3,torch::RestrictPtrTraits>(),   
    //             splatting_weights_tensor.packed_accessor32<float,3,torch::RestrictPtrTraits>(),   
    //             lattice_values_monolithic.packed_accessor32<float,3,torch::RestrictPtrTraits>()
    //         );
    //     }else{
    //         LOG(FATAL) <<"Instantiation for this val_dim doesnt exist. Please add it to the source code here.";
    //     }
    // }else{
    //     LOG(FATAL) <<"Instantiation for this pos_dim doesnt exist. Please add it to the source code here.";
    // }

    

    return lattice_values_monolithic;

}

std::tuple<torch::Tensor, torch::Tensor> Lattice::slice_backwards_standalone_no_precomputation_no_homogeneous_mr_monolithic(torch::Tensor& positions_raw,  torch::Tensor& lattice_values_monolithic, torch::Tensor& grad_sliced_values_monolithic, const torch::Tensor& scale_factor, torch::Tensor& random_shift_monolithic, torch::Tensor& anneal_window, const bool concat_points, const bool require_lattice_values_grad, const bool require_positions_grad){

    check_positions(positions_raw); 
    // CHECK(val_dim()>0) << "m_val_dim is 0 or lwoer. We have to splat something first so that we have values from where to slice. Val dim is " << val_dim();
    int nr_positions=positions_raw.size(0);
    int pos_dim=positions_raw.size(1);
    CHECK(pos_dim==this->pos_dim()) << " The position dimension do not coreespond with the ones we used for creating the lattice";
    CHECK(grad_sliced_values_monolithic.dim()==3) <<"grad_sliced_values_monolithic should be nr_resolutions x val_dim x nr_positions, so it should have 3 dimensions. However it has "<< grad_sliced_values_monolithic.dim();
    CHECK(grad_sliced_values_monolithic.is_contiguous()) << "Grad sliced values needs to be contiguous. Please call .contiguous() on it";
    // splatting_indices_tensor=splatting_indices_tensor.contiguous();
    // splatting_weights_tensor=splatting_weights_tensor.contiguous();
    int nr_resolutions=grad_sliced_values_monolithic.size(0);
    int val_dim=grad_sliced_values_monolithic.size(1);
    CHECK(nr_positions==grad_sliced_values_monolithic.size(2)) << "The nr of positions should match between the input positions and the sliced values";
    CHECK(lattice_values_monolithic.dim()==3) <<"grad_sliced_values_monolithic should be nr_resolutions x val_dim x nr_positions, so it should have 3 dimensions. However it has "<< lattice_values_monolithic.dim();
    CHECK(lattice_values_monolithic.is_contiguous()) <<"We assume that the lattice_values_monolithic are contiguous because in the cuda code we make a load of 2 float values at a time and that assumes that they are contiguous";
    

    //if we concat also the points, we add a series of extra resolutions to contain those points
    int nr_resolutions_extra=0;
    if (concat_points){
        nr_resolutions_extra=std::ceil(float(pos_dim)/val_dim);
        // VLOG(1) << "grad_sliced_values_monolithic" << grad_sliced_values_monolithic.sizes();
        // VLOG(1) << "grad_sliced_values_monolithic" << grad_sliced_values_monolithic;
        // grad_sliced_values_monolithic=grad_sliced_values_monolithic.slice(0, 0, nr_resolutions-nr_resolutions_extra); //dim, start, end
        // VLOG(1) << "grad_sliced_values_monolithic after slicing" << grad_sliced_values_monolithic.sizes();
        // VLOG(1) << "grad_sliced_values_monolithic after slicing" << grad_sliced_values_monolithic;
        nr_resolutions=nr_resolutions-nr_resolutions_extra;
    }

    

    // nr_resolutions x nr_lattice_vertices x nr_lattice_featues
    // Tensor lattice_values_monolithic=torch::zeros({ nr_resolutions, nr_lattice_vertices(), val_dim  },  torch::dtype(torch::kFloat32).device(torch::kCUDA, 0)  );
    // Tensor lattice_values_monolithic=torch::zeros({ nr_resolutions, val_dim, m_capacity },  torch::dtype(torch::kFloat32).device(torch::kCUDA, 0)  );
    // TIME_START("slice_b_create_output");
    Tensor lattice_values_monolithic_grad; //dL/dLattiveValues
    if (require_lattice_values_grad){
        #if LATTICE_HALF_PRECISION
            // Tensor lattice_values_monolithic=torch::zeros({ nr_resolutions, val_dim, m_capacity },  torch::dtype(torch::kFloat16).device(torch::kCUDA, 0)  );
            lattice_values_monolithic_grad=torch::zeros({ nr_resolutions, m_capacity, val_dim },  torch::dtype(torch::kFloat16).device(torch::kCUDA, 0)  );
        #else
            // Tensor lattice_values_monolithic=torch::zeros({ nr_resolutions, val_dim, m_capacity },  torch::dtype(torch::kFloat32).device(torch::kCUDA, 0)  );
            lattice_values_monolithic_grad=torch::zeros({ nr_resolutions, m_capacity, val_dim },  torch::dtype(torch::kFloat32).device(torch::kCUDA, 0)  );
        #endif
    }else{
        lattice_values_monolithic_grad=torch::empty({ 1,1,1 },  torch::dtype(torch::kFloat32).device(torch::kCUDA, 0)  );
    }
    


    Tensor positions_grad; //dL/dPos
    if (require_positions_grad){
        positions_grad=torch::zeros({ nr_positions, pos_dim },  torch::dtype(torch::kFloat32).device(torch::kCUDA, 0)  );
    }else{
        positions_grad=torch::empty({ 1,1 },  torch::dtype(torch::kFloat32).device(torch::kCUDA, 0)  );
    }

    // TIME_END("slice_b_create_output");

    //make scalefactor for each lvl
    // Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> scale_factor_eigen;
    // scale_factor_eigen.resize(nr_resolutions, pos_dim);
    // double invStdDev = 1.0;
    // for(int res_idx=0; res_idx<nr_resolutions; res_idx++){
    //     for (int i = 0; i < pos_dim; i++) {
    //         scale_factor_eigen(res_idx,i) =  1.0 / (std::sqrt((double) (i + 1) * (i + 2))) * invStdDev;
    //         scale_factor_eigen(res_idx,i)=scale_factor_eigen(res_idx,i)/ sigmas_list[res_idx];
    //         // VLOG(1) << "scalinbg by " << sigmas_list[res_idx];
    //     }
    // }
    // // VLOG(1) << "scale_factor_eigen" << scale_factor_eigen;
    // Tensor scale_factor_tensor=eigen2tensor(scale_factor_eigen.cast<float>()).cuda();
    // scale_factor_tensor=scale_factor_tensor.view({nr_resolutions, pos_dim}); //nr_resolutuons x pos_dim

    // Tensor scale_factor_tensor=Lattice::compute_scale_factor_tensor(sigmas_list, pos_dim);
    Tensor scale_factor_tensor=scale_factor;


    const dim3 blocks = { (unsigned int)div_round_up(nr_positions, BLOCK_SIZE_BACK), (unsigned int)nr_resolutions, 1 }; //the blocks are executed in order, first the blocks for the first resolution, then the second and so on

    // TIME_START("slice_back_cuda");
    if (pos_dim==2){
        if(val_dim==2){
            slice_backwards_no_precomputation_no_homogeneous_mr_monolithic<2,2><<<blocks, BLOCK_SIZE_BACK>>>(
                nr_positions,
                m_capacity, 
                lattice_values_monolithic.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
                positions_raw.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                scale_factor_tensor.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                random_shift_monolithic.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                anneal_window.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
                // grad_sliced_values_monolithic.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
                // #if LATTICE_HALF_PRECISION
                    // grad_sliced_values_monolithic.packed_accessor32<at::Half,3,torch::RestrictPtrTraits>(),
                // #else
                    grad_sliced_values_monolithic.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
                // #endif
                // lattice_values_monolithic.packed_accessor32<float,3,torch::RestrictPtrTraits>()
                #if LATTICE_HALF_PRECISION
                    lattice_values_monolithic_grad.packed_accessor32<at::Half,3,torch::RestrictPtrTraits>(),
                #else   
                    lattice_values_monolithic_grad.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
                #endif
                positions_grad.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                concat_points,
                require_lattice_values_grad,
                require_positions_grad
            );
        }else{
            LOG(FATAL) <<"Instantiation for this val_dim doesnt exist. Please add it to the source code here.";
        }
    }else if(pos_dim==3){
        // LOG(FATAL) <<"Instantiation for this pos_dim doesnt exist. Please add it to the source code here.";
        if(val_dim==2){
            slice_backwards_no_precomputation_no_homogeneous_mr_monolithic<3,2><<<blocks, BLOCK_SIZE_BACK>>>(
                nr_positions,
                m_capacity, 
                lattice_values_monolithic.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
                positions_raw.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                scale_factor_tensor.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                random_shift_monolithic.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                anneal_window.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
                // grad_sliced_values_monolithic.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
                // #if LATTICE_HALF_PRECISION
                    // grad_sliced_values_monolithic.packed_accessor32<at::Half,3,torch::RestrictPtrTraits>(),
                // #else
                    grad_sliced_values_monolithic.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
                // #endif
                // lattice_values_monolithic.packed_accessor32<float,3,torch::RestrictPtrTraits>()
                #if LATTICE_HALF_PRECISION
                    lattice_values_monolithic_grad.packed_accessor32<at::Half,3,torch::RestrictPtrTraits>(),
                #else   
                    lattice_values_monolithic_grad.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
                #endif
                positions_grad.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                concat_points,
                require_lattice_values_grad,
                require_positions_grad
                
            );
        }else{
            LOG(FATAL) << "I'll implement it later. For now ther are a lot of stuff in the kernel hard coded for valdim=2 and pos either 3 or 4";
        }
    }else if(pos_dim==4){
        // LOG(FATAL) <<"Instantiation for this pos_dim doesnt exist. Please add it to the source code here.";
        if(val_dim==2){
            slice_backwards_no_precomputation_no_homogeneous_mr_monolithic<4,2><<<blocks, BLOCK_SIZE_BACK>>>(
                nr_positions,
                m_capacity, 
                lattice_values_monolithic.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
                positions_raw.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                scale_factor_tensor.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                random_shift_monolithic.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                anneal_window.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
                // grad_sliced_values_monolithic.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
                // #if LATTICE_HALF_PRECISION
                    // grad_sliced_values_monolithic.packed_accessor32<at::Half,3,torch::RestrictPtrTraits>(),
                // #else
                    grad_sliced_values_monolithic.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
                // #endif
                // lattice_values_monolithic.packed_accessor32<float,3,torch::RestrictPtrTraits>()
                #if LATTICE_HALF_PRECISION
                    lattice_values_monolithic_grad.packed_accessor32<at::Half,3,torch::RestrictPtrTraits>(),
                #else   
                    lattice_values_monolithic_grad.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
                #endif
                positions_grad.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                concat_points,
                require_lattice_values_grad,
                require_positions_grad
                
            );
        }else{
            LOG(FATAL) << "I'll implement it later. For now ther are a lot of stuff in the kernel hard coded for valdim=2 and pos either 3 or 4";
        }
    }else if(pos_dim==5){
        // LOG(FATAL) <<"Instantiation for this pos_dim doesnt exist. Please add it to the source code here.";
        if(val_dim==2){
            slice_backwards_no_precomputation_no_homogeneous_mr_monolithic<5,2><<<blocks, BLOCK_SIZE_BACK>>>(
                nr_positions,
                m_capacity, 
                lattice_values_monolithic.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
                positions_raw.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                scale_factor_tensor.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                random_shift_monolithic.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                anneal_window.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
                // grad_sliced_values_monolithic.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
                // #if LATTICE_HALF_PRECISION
                    // grad_sliced_values_monolithic.packed_accessor32<at::Half,3,torch::RestrictPtrTraits>(),
                // #else
                    grad_sliced_values_monolithic.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
                // #endif
                // lattice_values_monolithic.packed_accessor32<float,3,torch::RestrictPtrTraits>()
                #if LATTICE_HALF_PRECISION
                    lattice_values_monolithic_grad.packed_accessor32<at::Half,3,torch::RestrictPtrTraits>(),
                #else   
                    lattice_values_monolithic_grad.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
                #endif
                positions_grad.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                concat_points,
                require_lattice_values_grad,
                require_positions_grad
                
            );
        }else{
            LOG(FATAL) << "I'll implement it later. For now ther are a lot of stuff in the kernel hard coded for valdim=2 and pos either 3 or 4";
        }
    }else if(pos_dim==6){
        // LOG(FATAL) <<"Instantiation for this pos_dim doesnt exist. Please add it to the source code here.";
        if(val_dim==2){
            slice_backwards_no_precomputation_no_homogeneous_mr_monolithic<6,2><<<blocks, BLOCK_SIZE_BACK>>>(
                nr_positions,
                m_capacity, 
                lattice_values_monolithic.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
                positions_raw.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                scale_factor_tensor.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                random_shift_monolithic.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                anneal_window.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
                // grad_sliced_values_monolithic.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
                // #if LATTICE_HALF_PRECISION
                    // grad_sliced_values_monolithic.packed_accessor32<at::Half,3,torch::RestrictPtrTraits>(),
                // #else
                    grad_sliced_values_monolithic.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
                // #endif
                // lattice_values_monolithic.packed_accessor32<float,3,torch::RestrictPtrTraits>()
                #if LATTICE_HALF_PRECISION
                    lattice_values_monolithic_grad.packed_accessor32<at::Half,3,torch::RestrictPtrTraits>(),
                #else   
                    lattice_values_monolithic_grad.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
                #endif
                positions_grad.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                concat_points,
                require_lattice_values_grad,
                require_positions_grad
                
            );
        }else{
            LOG(FATAL) << "I'll implement it later. For now ther are a lot of stuff in the kernel hard coded for valdim=2 and pos either 3 or 4";
        }
    }
    // TIME_END("slice_back_cuda");
    
   

    return std::make_tuple(lattice_values_monolithic_grad,positions_grad);

}


//double back
std::tuple<torch::Tensor, torch::Tensor> Lattice::slice_double_back_from_positions_grad(const torch::Tensor& double_positions_grad, torch::Tensor& positions_raw, torch::Tensor& lattice_values_monolithic, torch::Tensor& grad_sliced_values_monolithic, const torch::Tensor& scale_factor, torch::Tensor& random_shift_monolithic, torch::Tensor& anneal_window, const bool concat_points){

    check_positions(positions_raw); 
    // CHECK(val_dim()>0) << "m_val_dim is 0 or lwoer. We have to splat something first so that we have values from where to slice. Val dim is " << val_dim();
    int nr_positions=positions_raw.size(0);
    int pos_dim=positions_raw.size(1);
    CHECK(pos_dim==this->pos_dim()) << " The position dimension do not coreespond with the ones we used for creating the lattice";
    CHECK(grad_sliced_values_monolithic.dim()==3) <<"grad_sliced_values_monolithic should be nr_resolutions x val_dim x nr_positions, so it should have 3 dimensions. However it has "<< grad_sliced_values_monolithic.dim();
    CHECK(grad_sliced_values_monolithic.is_contiguous()) << "Grad sliced values needs to be contiguous. Please call .contiguous() on it";
    // splatting_indices_tensor=splatting_indices_tensor.contiguous();
    // splatting_weights_tensor=splatting_weights_tensor.contiguous();
    int nr_resolutions=grad_sliced_values_monolithic.size(0);
    int val_dim=grad_sliced_values_monolithic.size(1);
    CHECK(nr_positions==grad_sliced_values_monolithic.size(2)) << "The nr of positions should match between the input positions and the sliced values";
    CHECK(lattice_values_monolithic.dim()==3) <<"grad_sliced_values_monolithic should be nr_resolutions x val_dim x nr_positions, so it should have 3 dimensions. However it has "<< lattice_values_monolithic.dim();
    CHECK(lattice_values_monolithic.is_contiguous()) <<"We assume that the lattice_values_monolithic are contiguous because in the cuda code we make a load of 2 float values at a time and that assumes that they are contiguous";


    //if we concat also the points, we add a series of extra resolutions to contain those points
    int nr_resolutions_extra=0;
    if (concat_points){
        nr_resolutions_extra=std::ceil(float(pos_dim)/val_dim);
        // VLOG(1) << "grad_sliced_values_monolithic" << grad_sliced_values_monolithic.sizes();
        // VLOG(1) << "grad_sliced_values_monolithic" << grad_sliced_values_monolithic;
        // grad_sliced_values_monolithic=grad_sliced_values_monolithic.slice(0, 0, nr_resolutions-nr_resolutions_extra); //dim, start, end
        // VLOG(1) << "grad_sliced_values_monolithic after slicing" << grad_sliced_values_monolithic.sizes();
        // VLOG(1) << "grad_sliced_values_monolithic after slicing" << grad_sliced_values_monolithic;
        nr_resolutions=nr_resolutions-nr_resolutions_extra;
    }

    

    // nr_resolutions x nr_lattice_vertices x nr_lattice_featues
    // Tensor lattice_values_monolithic=torch::zeros({ nr_resolutions, nr_lattice_vertices(), val_dim  },  torch::dtype(torch::kFloat32).device(torch::kCUDA, 0)  );
    // Tensor lattice_values_monolithic=torch::zeros({ nr_resolutions, val_dim, m_capacity },  torch::dtype(torch::kFloat32).device(torch::kCUDA, 0)  );
    // TIME_START("slice_b_create_output");
    Tensor lattice_values_monolithic_grad; //dL/dLattiveValues
        #if LATTICE_HALF_PRECISION
            // Tensor lattice_values_monolithic=torch::zeros({ nr_resolutions, val_dim, m_capacity },  torch::dtype(torch::kFloat16).device(torch::kCUDA, 0)  );
            lattice_values_monolithic_grad=torch::zeros({ nr_resolutions, m_capacity, val_dim },  torch::dtype(torch::kFloat16).device(torch::kCUDA, 0)  );
        #else
            // Tensor lattice_values_monolithic=torch::zeros({ nr_resolutions, val_dim, m_capacity },  torch::dtype(torch::kFloat32).device(torch::kCUDA, 0)  );
            lattice_values_monolithic_grad=torch::zeros({ nr_resolutions, m_capacity, val_dim },  torch::dtype(torch::kFloat32).device(torch::kCUDA, 0)  );
        #endif


    Tensor grad_grad_sliced_values_monolithic = torch::zeros({ nr_resolutions+nr_resolutions_extra, val_dim, nr_positions },  torch::dtype(torch::kFloat32).device(torch::kCUDA, 0)  );
    


    
    Tensor scale_factor_tensor=scale_factor;


    const dim3 blocks = { (unsigned int)div_round_up(nr_positions, BLOCK_SIZE_BACK), (unsigned int)nr_resolutions, 1 }; //the blocks are executed in order, first the blocks for the first resolution, then the second and so on

    // TIME_START("slice_double_back");
    if (pos_dim==3){
        if(val_dim==1){
            // slice_backwards_no_precomputation_no_homogeneous_mr_monolithic<3,1><<<blocks, BLOCK_SIZE_BACK>>>(
            //     nr_positions, 
            //     m_capacity,
            //     positions_raw.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
            //     scale_factor_tensor.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
            //     grad_sliced_values_monolithic.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
            //     lattice_values_monolithic.packed_accessor32<float,3,torch::RestrictPtrTraits>()
            // );
            LOG(FATAL) << "I'll implement it later. For now ther are a lot of stuff in the kernel hard coded for valdim=2 and pos either 3 or 4";
        }else if(val_dim==2){
            // slice_backwards_no_precomputation_no_homogeneous_mr_monolithic<3,2><<<blocks, BLOCK_SIZE_BACK>>>(
            //     nr_positions, 
            //     m_capacity,
            //     positions_raw.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
            //     scale_factor_tensor.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
            //     grad_sliced_values_monolithic.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
            //     lattice_values_monolithic.packed_accessor32<float,3,torch::RestrictPtrTraits>()
            // );
            slice_double_back_from_positions_grad_gpu<3,2><<<blocks, BLOCK_SIZE_BACK>>>(
                nr_positions,
                m_capacity, 
                double_positions_grad.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                lattice_values_monolithic.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
                positions_raw.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                scale_factor_tensor.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                random_shift_monolithic.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
                anneal_window.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
                grad_sliced_values_monolithic.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
                concat_points,
                //output
                grad_grad_sliced_values_monolithic.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
                #if LATTICE_HALF_PRECISION
                    lattice_values_monolithic_grad.packed_accessor32<at::Half,3,torch::RestrictPtrTraits>()
                #else   
                    lattice_values_monolithic_grad.packed_accessor32<float,3,torch::RestrictPtrTraits>()
                #endif
            );
        }else if(val_dim==4){
            // slice_backwards_no_precomputation_no_homogeneous_mr_monolithic<3,4><<<blocks, BLOCK_SIZE_BACK>>>(
            //     nr_positions, 
            //     m_capacity,
            //     positions_raw.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
            //     scale_factor_tensor.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
            //     grad_sliced_values_monolithic.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
            //     lattice_values_monolithic.packed_accessor32<float,3,torch::RestrictPtrTraits>()
            // );
            LOG(FATAL) << "I'll implement it later. For now ther are a lot of stuff in the kernel hard coded for valdim=2 and pos either 3 or 4";
        }else{
            LOG(FATAL) <<"Instantiation for this val_dim doesnt exist. Please add it to the source code here.";
        }
    }else{
        LOG(FATAL) <<"Instantiation for this pos_dim doesnt exist. Please add it to the source code here.";
    }
    // TIME_END("slice_double_back");
    
   

    return std::make_tuple(lattice_values_monolithic_grad,  grad_grad_sliced_values_monolithic);

}


// torch::Tensor Lattice::elevate_points(torch::Tensor& positions_raw){

//     int nr_positions=positions_raw.size(0);
//     int pos_dim=positions_raw.size(1);

//     //to cuda
//     TIME_START("upload_cuda");
//     positions_raw=positions_raw.to("cuda");
//     m_sigmas_tensor=m_sigmas_tensor.to("cuda");
//     TIME_END("upload_cuda");

//     TIME_START("scale_by_sigma");
//     Tensor positions=positions_raw/m_sigmas_tensor;
//     TIME_END("scale_by_sigma");

//     Tensor elevated=torch::zeros({nr_positions, pos_dim+1}, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0) );
//     elevated.fill_(0);

//     TIME_START("elevate");
//     // m_impl->elevate(positions.data_ptr<float>(),  m_pos_dim, nr_positions, elevated.data_ptr<float>());


//     dim3 blocks((nr_positions - 1) / BLOCK_SIZE + 1, 1, 1);
//     dim3 blockSize(BLOCK_SIZE, 1, 1);
//     blockSize.y = 1;

//     if (pos_dim==3){
//         elevate_points_gpu<3><<<blocks, blockSize>>>(
//             nr_positions, 
//             positions.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
//             elevated.packed_accessor32<float,2,torch::RestrictPtrTraits>()
//         );
//     }else{
//         LOG(FATAL) <<"Instantiation for this pos_dim doesnt exist. Please add it to the source code here.";
//     }

//     TIME_END("elevate");

//     // elevated=elevated.unsqueeze(0);
//     // EigenMatrixXfRowMajor elevated_eigen_rowmajor=tensor2eigen(elevated);
//     // Eigen::MatrixXd elevated_eigen;
//     // elevated_eigen=elevated_eigen_rowmajor.cast<double>();
//     // return elevated_eigen;

//     return elevated;
// }



std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> Lattice::create_non_differentiable_indices_for_slice_with_collisions(torch::Tensor& positions_elevated){

    check_positions_elevated(positions_elevated); 
    // CHECK(val_dim()>0) << "m_val_dim is 0 or lwoer. We have to splat something first so that we have values from where to slice. Val dim is " << val_dim();
    int nr_positions=positions_elevated.size(0);
    int pos_dim=positions_elevated.size(1) -1 ;
    CHECK(pos_dim==this->pos_dim()) << " The position dimension do not coreespond with the ones we used for creating the lattice";


     //to cuda
    // TIME_START("upload_cuda");
    positions_elevated=positions_elevated.to("cuda");
    m_sigmas_tensor=m_sigmas_tensor.to("cuda");
    // TIME_END("upload_cuda");

    // TIME_START("scale_by_sigma");
    VLOG(3) << "slice standalone scaling by a sigma of " << m_sigmas_tensor;
    // Tensor positions=positions_raw/m_sigmas_tensor;
    // TIME_END("scale_by_sigma")

    //initialize the output values to zero 
    Tensor rem0_matrix=torch::zeros({nr_positions, this->pos_dim()+1 }, torch::dtype(torch::kInt32).device(torch::kCUDA, 0) );
    Tensor rank_matrix=torch::zeros({nr_positions, this->pos_dim()+1 }, torch::dtype(torch::kInt32).device(torch::kCUDA, 0) );
    Tensor splatting_indices_tensor = torch::empty({ (pos_dim+1)* nr_positions }, torch::dtype(torch::kInt32).device(torch::kCUDA, 0) );
    rem0_matrix.fill_(0);
    rank_matrix.fill_(0);
    splatting_indices_tensor.fill_(-1);


    dim3 blocks((nr_positions - 1) / BLOCK_SIZE + 1, 1, 1);
    dim3 blockSize(BLOCK_SIZE, 1, 1);
    blockSize.y = 1;

    if (pos_dim==3){
        create_non_differentiable_indices_for_slice_with_collisions_gpu<3><<<blocks, blockSize>>>(
            nr_positions, 
            m_capacity,
            positions_elevated.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
            rem0_matrix.packed_accessor32<int,2,torch::RestrictPtrTraits>(),   
            rank_matrix.packed_accessor32<int,2,torch::RestrictPtrTraits>(),   
            splatting_indices_tensor.packed_accessor32<int,1,torch::RestrictPtrTraits>()
            // *(m_hash_table->m_impl)
        );
    }else{
        LOG(FATAL) <<"Instantiation for this pos_dim doesnt exist. Please add it to the source code here.";
    }

    auto ret = std::make_tuple (rem0_matrix, rank_matrix, splatting_indices_tensor); 
    return ret;

}
// void Lattice::debug_repeted_weights(const Eigen::MatrixXf& points, const Eigen::MatrixXi& splatting_indices, const Eigen::MatrixXf& splatting_weights ){
//     int nr_points=points.rows();

//     Eigen::MatrixXi nr_times_weights_repeted;
//     nr_times_weights_repeted.resize(nr_points,4);
//     nr_times_weights_repeted.setZero();

//     //find how many times each splatting weight is repeted
//     for(int i=0; i<nr_points; i++){
//         for(int j=0; j<4; j++){ 
//             float cur_w=splatting_weights(i,j);
//             //for this cur_w go again through the whole vector and check if there is a repeted one
//             for(int ni=0; ni<nr_points; ni++){
//                 for(int nj=0; nj<4; nj++){ 
//                     float nw=splatting_weights(ni,nj);
//                     if(cur_w==nw){
//                         nr_times_weights_repeted(i,j)++;
//                         if(nr_times_weights_repeted(i,j)>5){
//                             VLOG(1) << "Index " <<i << " " << j << " repeted "<< nr_times_weights_repeted(i,j);
//                         }
//                     }
//                 }
//             }



//         }
//     }

// }


// std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> Lattice::create_non_differentiable_indices_for_slice_with_collisions(torch::Tensor& positions_raw){

//     check_positions(positions_raw); 
//     CHECK(val_dim()>0) << "m_val_dim is 0 or lwoer. We have to splat something first so that we have values from where to slice. Val dim is " << val_dim();
//     int nr_positions=positions_raw.size(0);
//     int pos_dim=positions_raw.size(1);
//     CHECK(pos_dim==this->pos_dim()) << " The position dimension do not coreespond with the ones we used for creating the lattice";


//      //to cuda
//     TIME_START("upload_cuda");
//     positions_raw=positions_raw.to("cuda");
//     m_sigmas_tensor=m_sigmas_tensor.to("cuda");
//     TIME_END("upload_cuda");

//     TIME_START("scale_by_sigma");
//     VLOG(3) << "slice standalone scaling by a sigma of " << m_sigmas_tensor;
//     Tensor positions=positions_raw/m_sigmas_tensor;
//     TIME_END("scale_by_sigma")

//     //initialize the output values to zero 
//     Tensor rem0_matrix=torch::zeros({nr_positions, this->pos_dim()+1 }, torch::dtype(torch::kInt32).device(torch::kCUDA, 0) );
//     Tensor rank_matrix=torch::zeros({nr_positions, this->pos_dim()+1 }, torch::dtype(torch::kInt32).device(torch::kCUDA, 0) );
//     Tensor splatting_indices_tensor = torch::empty({ (pos_dim+1)* nr_positions }, torch::dtype(torch::kInt32).device(torch::kCUDA, 0) );
//     rem0_matrix.fill_(0);
//     rank_matrix.fill_(0);
//     splatting_indices_tensor.fill_(-1);


//     dim3 blocks((nr_positions - 1) / BLOCK_SIZE + 1, 1, 1);
//     dim3 blockSize(BLOCK_SIZE, 1, 1);
//     blockSize.y = 1;

//     if (pos_dim==3){
//         create_non_differentiable_indices_for_slice_with_collisions_gpu<3><<<blocks, blockSize>>>(
//             nr_positions, 
//             positions.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
//             rem0_matrix.packed_accessor32<int,2,torch::RestrictPtrTraits>(),   
//             rank_matrix.packed_accessor32<int,2,torch::RestrictPtrTraits>(),   
//             splatting_indices_tensor.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
//             *(m_hash_table->m_impl)
//         );
//     }else{
//         LOG(FATAL) <<"Instantiation for this pos_dim doesnt exist. Please add it to the source code here.";
//     }

//     auto ret = std::make_tuple (rem0_matrix, rank_matrix, splatting_indices_tensor); 
//     return ret;

// }



// Eigen::MatrixXf Lattice::create_E_matrix(const int pos_dim){


//     //page 30 of Andrew Adams thesis
//     Eigen::MatrixXf E_left(pos_dim+1, pos_dim );
//     Eigen::MatrixXf E_right(pos_dim, pos_dim );
//     E_left.setZero();
//     E_right.setZero();
//     //E left is has at the bottom a square matrix which has an upper triangular part of ones. Afterwards the whole E_left gets appended another row on top of all ones
//     E_left.bottomRows(pos_dim).triangularView<Eigen::Upper>().setOnes();
//     //the diagonal of the bottom square is linearly incresing from [-1, -m_pos_dim]
//     E_left.bottomRows(pos_dim).diagonal().setLinSpaced(pos_dim,1,pos_dim);
//     E_left.bottomRows(pos_dim).diagonal()= -E_left.bottomRows(pos_dim).diagonal();
//     //E_left has the first row all set to ones
//     E_left.row(0).setOnes();
//     // VLOG(1) << "E left is \n" << E_left;
//     //E right is just a diagonal matrix with entried in the diag set to 1/sqrt((d+1)(d+2)). Take into account that the d in the paper starts at 1 and we start at 0 so we add a +1 to diag_idx
//     for(int diag_idx=0; diag_idx<pos_dim; diag_idx++){
//         E_right(diag_idx, diag_idx) =  1.0 / (sqrt((diag_idx + 1) * (diag_idx + 2))) ;
//         // VLOG(1) << "placing in the diagonal a "<< ((diag_idx + 1) * (diag_idx + 2));
//     }
//     // VLOG(1) << "E right is \n" << E_right;

//     //rotate into H_d
//     Eigen::MatrixXf E = E_left*E_right;

//     return E;
// }






// void Lattice::begin_splat(const bool reset_hashmap ){
//     // m_hash_table->clear(); 
//     if(reset_hashmap)   {
//         m_hash_table->clear(); 
//     }else {
//         m_hash_table->clear_only_values();
//     }
//     m_hash_table->m_nr_filled_is_dirty=true;
// }


// std::tuple<torch::Tensor, torch::Tensor> Lattice::splat_standalone(torch::Tensor& positions_raw, torch::Tensor& values ){
//     check_positions_and_values(positions_raw, values);
//     int nr_positions=positions_raw.size(0);
//     int pos_dim=positions_raw.size(1);
//     int val_dim=values.size(1);

//     m_positions=positions_raw; //raw positions which created this lattice


//     //if it's not initialized to the correct values we intialize the hashtable
//     if( !m_hash_table->m_keys_tensor.defined() ){
//         m_hash_table->init(pos_dim, val_dim);
//         m_hash_table->to(torch::kCUDA);
//     }

//     // if( !m_splatting_indices_tensor.defined() || m_splatting_indices_tensor.size(0)!=nr_positions*(m_pos_dim+1)  ){
//     Tensor splatting_indices_tensor = torch::empty({nr_positions*(pos_dim+1) }, torch::dtype(torch::kInt32).device(torch::kCUDA, 0) );
//     Tensor splatting_weights_tensor = torch::empty({nr_positions*(pos_dim+1) }, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0) );
//     splatting_indices_tensor.fill_(-1);
//     splatting_weights_tensor.fill_(-1);


//     //to cuda
//     TIME_START("upload_cuda");
//     positions_raw=positions_raw.to("cuda");
//     values=values.to("cuda");
//     m_sigmas_tensor=m_sigmas_tensor.to("cuda");
//     TIME_END("upload_cuda");

//     TIME_START("scale_by_sigma");
//     Tensor positions=positions_raw/m_sigmas_tensor;
//     TIME_END("scale_by_sigma");

//     TIME_START("splat");
//     m_impl->splat_standalone(positions.data_ptr<float>(), values.data_ptr<float>(), nr_positions, pos_dim, val_dim, 
//                             splatting_indices_tensor.data_ptr<int>(), splatting_weights_tensor.data_ptr<float>(),  *(m_hash_table->m_impl) );
//     m_hash_table->m_nr_filled_is_dirty=true;

    
//     TIME_END("splat");

//     // VLOG(3) << "after splatting nr_verts is " << nr_lattice_vertices();
//     auto ret = std::make_tuple (splatting_indices_tensor, splatting_weights_tensor ); 
//     return ret;
  
// }


// std::tuple<torch::Tensor, torch::Tensor> Lattice::just_create_verts(torch::Tensor& positions_raw, const bool return_indices_and_weights ){
//     check_positions(positions_raw);
//     int nr_positions=positions_raw.size(0);
//     int pos_dim=positions_raw.size(1);

//     //if it's not initialized to the correct values we intialize the hashtable
//     if( !m_hash_table->m_keys_tensor.defined() ){
//         m_hash_table->init(pos_dim, 1 );
//         m_hash_table->to(torch::kCUDA);
//     }


//     Tensor splatting_indices_tensor;
//     Tensor splatting_weights_tensor;
//     if (return_indices_and_weights){
//         splatting_indices_tensor = torch::empty({nr_positions*(pos_dim+1) }, torch::dtype(torch::kInt32).device(torch::kCUDA, 0) );
//         splatting_weights_tensor = torch::empty({nr_positions*(pos_dim+1) }, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0) );
//         splatting_indices_tensor.fill_(-1);
//         splatting_weights_tensor.fill_(-1);
//     }


//     //to cuda
//     positions_raw=positions_raw.to("cuda");
//     m_sigmas_tensor=m_sigmas_tensor.to("cuda");

//     Tensor positions=positions_raw/m_sigmas_tensor;

//     if (return_indices_and_weights){
//         m_impl->just_create_verts(positions.data_ptr<float>(), nr_positions, this->pos_dim(), this->val_dim(), 
//                                 return_indices_and_weights,
//                                 splatting_indices_tensor.data_ptr<int>(), splatting_weights_tensor.data_ptr<float>(), *(m_hash_table->m_impl) );
//     }else{ 
//         m_impl->just_create_verts(positions.data_ptr<float>(), nr_positions, this->pos_dim(), this->val_dim(), 
//                                 return_indices_and_weights,
//                                 nullptr, nullptr, *(m_hash_table->m_impl) );
//     }
    
//     m_hash_table->m_nr_filled_is_dirty=true;


//     // VLOG(3) << "after just_create_verts nr_verts is " << nr_lattice_vertices();

//     auto ret = std::make_tuple (splatting_indices_tensor, splatting_weights_tensor ); 
//     return ret;
  
// }

// std::shared_ptr<Lattice> Lattice::expand(torch::Tensor& positions_raw, const int point_multiplier, const float noise_stddev, const bool expand_values ){
//     check_positions(positions_raw);
//     int pos_dim=positions_raw.size(1);

//     //if it's not initialized to the correct values we intialize the hashtable
//     if( !m_hash_table->m_keys_tensor.defined() ){
//         m_hash_table->init(pos_dim, 1 );
//         m_hash_table->to(torch::kCUDA);
//     }

//     //to cuda
//     positions_raw=positions_raw.to("cuda");
//     m_sigmas_tensor=m_sigmas_tensor.to("cuda");

//     //expand the positopns
//     Tensor positions_expanded=positions_raw.repeat({point_multiplier, 1});

//     //noise 
//     Tensor noise = torch::randn({ positions_expanded.size(0), positions_expanded.size(1) }, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0) );
//     noise=noise*noise_stddev;
//     positions_expanded+=noise;


//     std::shared_ptr<Lattice> expanded_lattice=create(this); //create a lattice with no config but takes the config from this one
//     expanded_lattice->m_name="expanded_lattice";
//     expanded_lattice->m_hash_table->m_values_tensor=torch::zeros({1, this->val_dim()}, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0) ); //we just create some dummy values just so that the clear that we will do not will not destroy the current values. We will create the values when we know how many vertices we have
//     expanded_lattice->m_hash_table->m_keys_tensor= this->m_hash_table->m_keys_tensor.clone();
//     expanded_lattice->m_hash_table->m_entries_tensor= this->m_hash_table->m_entries_tensor.clone();
//     expanded_lattice->m_hash_table->m_nr_filled_tensor= this->m_hash_table->m_nr_filled_tensor.clone();
//     expanded_lattice->m_hash_table->update_impl();




//     // int nr_positions=positions_expanded.size(0);
//     expanded_lattice->just_create_verts(positions_expanded, false );
//     expanded_lattice->m_hash_table->m_nr_filled_is_dirty=true;


//     if (expand_values){
//         int nr_values_diff= expanded_lattice->nr_lattice_vertices() - nr_lattice_vertices();
//         CHECK(nr_values_diff>=0) << "Nr of values in the difference is negative, we should always create more vertices, never substract so this doesnt make sense. In the current lattice we have " << nr_lattice_vertices() << " and in the expanded one we have " <<  expanded_lattice->nr_lattice_vertices();

//         std::vector<int64_t> pad_values { 0,0,0,nr_values_diff  }; //left, right, top, bottom
//         torch::nn::functional::PadFuncOptions option (pad_values);
//         option.mode(torch::kConstant);
//         Tensor expanded_values = torch::nn::functional::pad( values(), option);
//         expanded_lattice->set_values(expanded_values);

//         CHECK(expanded_lattice->values().size(0) == expanded_lattice->nr_lattice_vertices() ) << "The nr of lattice vertices and the nr of rows in the values should be the same. However we have nr of vertices " << expanded_lattice->nr_lattice_vertices() << " and the values have nr of rows " <<expanded_lattice->values().size(0);

//     }


//     return expanded_lattice;

// }


// std::tuple<std::shared_ptr<Lattice>, torch::Tensor, torch::Tensor, torch::Tensor> Lattice::distribute(torch::Tensor& positions_raw, torch::Tensor& values, const bool reset_hashmap){
//     check_positions_and_values(positions_raw, values);
//     int nr_positions=positions_raw.size(0);
//     int pos_dim=positions_raw.size(1);
//     int val_dim=values.size(1);

//     m_positions=positions_raw; //raw positions which created this lattice


    
//     //if it's not initialized to the correct values we intialize the hashtable
//     if(!m_hash_table->m_keys_tensor.defined()){
//         m_hash_table->init(pos_dim, val_dim);
//         // m_hash_table->to(torch::kCUDA);
//     }


//     Tensor distributed_tensor = torch::zeros({ nr_positions *(pos_dim+1) , pos_dim + val_dim +1 }, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0) );

//     Tensor splatting_indices_tensor = torch::empty({nr_positions*(pos_dim+1) }, torch::dtype(torch::kInt32).device(torch::kCUDA, 0) );
//     Tensor splatting_weights_tensor = torch::empty({nr_positions*(pos_dim+1) }, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0) );
//     splatting_indices_tensor.fill_(-1);
//     splatting_weights_tensor.fill_(-1);


//     std::shared_ptr<Lattice> distributed_lattice=create(this); //create a lattice with no config but takes the config from this one
//     distributed_lattice->m_hash_table->m_keys_tensor=this->m_hash_table->m_keys_tensor.clone();
//     distributed_lattice->m_hash_table->m_entries_tensor=this->m_hash_table->m_entries_tensor.clone();
//     if ( this->m_hash_table->m_values_tensor.defined()){
//         distributed_lattice->m_hash_table->m_values_tensor=this->m_hash_table->m_values_tensor.clone();
//     }
//     distributed_lattice->m_name="distributed_lattice";
//     distributed_lattice->m_hash_table->update_impl();
   
    
//     // m_hash_table->clear();
//     if(reset_hashmap)   {
//         distributed_lattice->m_hash_table->clear(); 
//     }else {
//         distributed_lattice->m_hash_table->clear_only_values();
//     }


//     //to cuda
//     positions_raw=positions_raw.to("cuda");
//     values=values.to("cuda");
//     m_sigmas_tensor=m_sigmas_tensor.to("cuda");

//     Tensor positions=positions_raw/m_sigmas_tensor;

//     m_impl->distribute(positions.data_ptr<float>(), values.data_ptr<float>(), distributed_tensor.data_ptr<float>(), nr_positions, pos_dim, val_dim, 
//                             splatting_indices_tensor.data_ptr<int>(), splatting_weights_tensor.data_ptr<float>(), *(distributed_lattice->m_hash_table->m_impl) );
//     distributed_lattice->m_hash_table->m_nr_filled_is_dirty=true;

//     VLOG(3) << "after distributing nr_verts is " << distributed_lattice->nr_lattice_vertices();

//     auto ret = std::make_tuple (distributed_lattice, distributed_tensor, splatting_indices_tensor, splatting_weights_tensor ); 
//     return ret;
  
// }






// std::shared_ptr<Lattice> Lattice::convolve_im2row_standalone(torch::Tensor& filter_bank, const int dilation, std::shared_ptr<Lattice> lattice_neighbours,  const bool flip_neighbours){

//     if (!lattice_neighbours){
//         lattice_neighbours=shared_from_this();
//     }

//     CHECK(filter_bank.defined()) << "Filter bank is undefined";
//     CHECK(filter_bank.dim()==2) << "Filter bank should have dimension 2, corresponding with (filter_extent * val_dim) x nr_filters.  However it has dimension: " << filter_bank.dim();
//     filter_bank=filter_bank.contiguous();

//     int nr_filters=filter_bank.size(1) ;
//     int filter_extent=filter_bank.size(0) / lattice_neighbours->val_dim();
//     CHECK(filter_extent == get_filter_extent(1) ) << "Filters should convolve over all the neighbours in the 1 hop plus the center vertex lattice. So the filter extent should be " << get_filter_extent(1) << ". However it is" << filter_extent << "val dim is " << lattice_neighbours->val_dim();

//     //this lattice should be coarser (so a higher lvl) or finer(lower lvl) or at least at the same lvl as the lattice neigbhours. But the differnce should be at most 1 level
//     CHECK(std::abs(m_lvl-lattice_neighbours->m_lvl)<=1) << "the difference in levels between query and neigbhours lattice should be only 1 or zero, so the query should be corser by 1 level or finer by 1 lvl with respect to the neighbours. Or if they are at the same level then nothing needs to be done. However the current lattice lvl is " << m_lvl << " and the neighbours lvl is " << lattice_neighbours->m_lvl;
    
//     // VLOG(4) <<"starting convolved im2row_standlaone. The current lattice has nr_vertices_lattices" << nr_lattice_vertices();
//     CHECK(nr_lattice_vertices()!=0) << "Why does this current lattice have zero nr_filled?";
//     int nr_vertices=nr_lattice_vertices();
//     int cur_values_size=m_hash_table->m_values_tensor.size(0);
//     // CHECK(nr_vertices==cur_values_size) << "the nr of lattice vertices should be the same as the values tensor has rows. However the nr lattice vertices is " << nr_vertices << " and values has nr of rows " << cur_values_size;

//     std::shared_ptr<Lattice> convolved_lattice=create(this); //create a lattice with no config but takes the config from this one
//     convolved_lattice->m_name="convolved_lattice";


//     filter_bank=filter_bank.to("cuda");


//     Tensor lattice_rowified=torch::zeros({nr_vertices, filter_extent* lattice_neighbours->val_dim() }, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0) );

    
//     m_impl->im2row(nr_vertices, this->pos_dim(), lattice_neighbours->val_dim(), dilation, lattice_rowified.data_ptr<float>(), filter_extent, *(m_hash_table->m_impl), *(lattice_neighbours->m_hash_table->m_impl), m_lvl, lattice_neighbours->m_lvl, flip_neighbours, false);
    


//     //multiply each patch with the filter bank
//     Tensor convolved= lattice_rowified.mm(filter_bank);
   
//     convolved_lattice->m_hash_table->set_values(convolved);
//     // convolved_lattice->m_val_dim=nr_filters;
//     // convolved_lattice->m_hash_table->update_impl(); //very important
//     // convolved_lattice->m_lattice_rowified=m_lattice_rowified;

//     // VLOG(4) << "convolved lattice has nr filled " << convolved_lattice->nr_lattice_vertices();
//     CHECK(convolved_lattice->nr_lattice_vertices()!=0) << "Why does this convolved lattice has zero nr_filled?";

//     return convolved_lattice;

// }

// torch::Tensor Lattice::im2row(std::shared_ptr<Lattice> lattice_neighbours, const int filter_extent, const int dilation, const bool flip_neighbours){

//     if (!lattice_neighbours){
//         lattice_neighbours=shared_from_this();
//     }

//     CHECK(filter_extent == get_filter_extent(1) ) << "Filters should convolve over all the neighbours in the 1 hop plus the center vertex lattice. So the filter extent should be " << get_filter_extent(1) << ". However it is" << filter_extent;


    
//     //this lattice should be coarser (so a higher lvl) or finer(lower lvl) or at least at the same lvl as the lattice neigbhours. But the differnce should be at most 1 level
//     CHECK(std::abs(m_lvl-lattice_neighbours->m_lvl)<=1) << "the difference in levels between query and neigbhours lattice should be only 1 or zero, so the query should be corser by 1 level or finer by 1 lvl with respect to the neighbours. Or if they are at the same level then nothing needs to be done. However the current lattice lvl is " << m_lvl << " and the neighbours lvl is " << lattice_neighbours->m_lvl;

//     // VLOG(3) <<"starting convolved im2row_standlaone. The current lattice has nr_vertices_lattices" << nr_lattice_vertices();
//     CHECK(nr_lattice_vertices()!=0) << "Why does this current lattice have zero nr_filled?";
//     int nr_vertices=nr_lattice_vertices();
//     int cur_values_size=m_hash_table->m_values_tensor.size(0);
//     // CHECK(nr_vertices==cur_values_size) << "the nr of lattice vertices should be the same as the values tensor has rows. However the nr lattice vertices is " << nr_vertices << " and values has nr of rows " << cur_values_size;


//     // TIME_START("create_lattice_rowified");
//     // if( !m_lattice_rowified.defined() || m_lattice_rowified.size(0)!=nr_vertices || m_lattice_rowified.size(1)!=filter_extent*m_val_dim  ){
//     Tensor lattice_rowified=torch::zeros({nr_vertices, filter_extent* lattice_neighbours->val_dim() }, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0) );
//     // }else{
//         // m_lattice_rowified.fill_(0);
//     // }


//     m_impl->im2row(nr_vertices, this->pos_dim(), lattice_neighbours->val_dim(), dilation, lattice_rowified.data_ptr<float>(), filter_extent, *(m_hash_table->m_impl), *(lattice_neighbours->m_hash_table->m_impl), m_lvl, lattice_neighbours->m_lvl, flip_neighbours, false);

//     return lattice_rowified;

// }

// torch::Tensor Lattice::row2im(const torch::Tensor& lattice_rowified,  const int dilation, const int filter_extent, const int nr_filters, std::shared_ptr<Lattice> lattice_neighbours){

//     CHECK(lattice_rowified.is_contiguous()) << "lattice rowified is not contiguous. Please call .contiguous() on it";
//     CHECK(lattice_rowified.size(1)/filter_extent == val_dim() ) << "Each row of the lattice rowified shold be of size val_dim*filter_extent. But the row size is " << lattice_rowified.size(1) << " and th val dim is " << val_dim();

//     if (!lattice_neighbours){
//         lattice_neighbours=shared_from_this();
//     }

//     int nr_vertices=nr_lattice_vertices();
//     m_hash_table->m_values_tensor=torch::zeros({nr_vertices, val_dim() }, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0) ); 
//     m_hash_table->update_impl();

//     CHECK(nr_lattice_vertices()!=0) <<"Something went wrong because have zero lattice vertices";


//     // m_val_dim=nr_filters;

//     m_impl->row2im(m_hash_table->capacity(), this->pos_dim(), lattice_neighbours->val_dim(), dilation, lattice_rowified.data_ptr<float>(), filter_extent, *(m_hash_table->m_impl), *(lattice_neighbours->m_hash_table->m_impl), m_lvl, lattice_neighbours->m_lvl, /*do_test*/false);

//     return m_hash_table->m_values_tensor;
// }


// std::shared_ptr<Lattice> Lattice::create_coarse_verts(){

//     int capacity=m_hash_table->capacity();
//     int val_dim=m_hash_table->val_dim();
//     int pos_dim=m_hash_table->pos_dim();

//     std::shared_ptr<Lattice> coarse_lattice=create(this); //create a lattice with no config but takes the config from this one
//     coarse_lattice->m_name="coarse_lattice";
//     coarse_lattice->m_lvl=m_lvl+1;
//     coarse_lattice->m_sigmas_tensor=m_sigmas_tensor.clone()*2.0; //the sigma for the coarser one is double. This is done so if we slice at this lattice we scale the positions with the correct sigma
//     for(size_t i=0; i<m_sigmas.size(); i++){
//         coarse_lattice->m_sigmas[i]=m_sigmas[i]*2.0;
//     } 
//     coarse_lattice->m_hash_table->m_values_tensor=torch::zeros({1, val_dim }, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0) ); //we just create some dummy values just so that the clear that we will do not will not destroy the current values. We will create the values when we know how many vertices we have
//     coarse_lattice->m_hash_table->m_keys_tensor=torch::zeros({capacity, pos_dim}, torch::dtype(torch::kInt32).device(torch::kCUDA, 0) );
//     coarse_lattice->m_hash_table->m_entries_tensor=torch::zeros({capacity}, torch::dtype(torch::kInt32).device(torch::kCUDA, 0) ) ;
//     coarse_lattice->m_hash_table->m_nr_filled_tensor=torch::zeros({1}, torch::dtype(torch::kInt32).device(torch::kCUDA, 0) );
//     coarse_lattice->m_hash_table->m_nr_filled_is_dirty=true;
//     coarse_lattice->m_hash_table->clear();
//     coarse_lattice->m_hash_table->update_impl();

//     TIME_START("coarsen");
//     m_impl->coarsen(capacity, pos_dim, *(m_hash_table->m_impl), *(coarse_lattice->m_hash_table->m_impl)  );
//     TIME_END("coarsen");

//     int nr_vertices=coarse_lattice->nr_lattice_vertices();
//     VLOG(3) << "after coarsening nr_verts of the coarse lattice is " << nr_vertices;

//     coarse_lattice->m_hash_table->m_values_tensor=torch::zeros({nr_vertices, val_dim}, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0)  ); //we create exactly the values required for he vertices that were allocated
//     coarse_lattice->m_hash_table->update_impl();

//     return coarse_lattice;

// }


// std::shared_ptr<Lattice> Lattice::create_coarse_verts_naive(torch::Tensor& positions_raw){

//     check_positions(positions_raw);

//     int capacity=m_hash_table->capacity();
//     int val_dim=m_hash_table->val_dim();
//     int pos_dim=m_hash_table->pos_dim();


//     std::shared_ptr<Lattice> coarse_lattice=create(this); //create a lattice with no config but takes the config from this one
//     coarse_lattice->m_name="coarse_lattice";
//     coarse_lattice->m_lvl=m_lvl+1;
//     coarse_lattice->m_sigmas_tensor=m_sigmas_tensor.clone()*2.0; //the sigma for the coarser one is double. This is done so if we slice at this lattice we scale the positions with the correct sigma
//     coarse_lattice->m_sigmas=m_sigmas;
//     for(size_t i=0; i<m_sigmas.size(); i++){
//         coarse_lattice->m_sigmas[i]=m_sigmas[i]*2.0;
//     } 
//     coarse_lattice->m_hash_table->m_values_tensor=torch::zeros({1, val_dim}, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0) ); //we just create some dummy values just so that the clear that we will do not will not destroy the current values. We will create the values when we know how many vertices we have
//     coarse_lattice->m_hash_table->m_keys_tensor=torch::zeros({capacity, pos_dim}, torch::dtype(torch::kInt32).device(torch::kCUDA, 0) );
//     coarse_lattice->m_hash_table->m_entries_tensor=torch::zeros({capacity}, torch::dtype(torch::kInt32).device(torch::kCUDA, 0) ) ;
//     coarse_lattice->m_hash_table->m_nr_filled_tensor=torch::zeros({1}, torch::dtype(torch::kInt32).device(torch::kCUDA, 0) );
//     coarse_lattice->m_hash_table->m_nr_filled_is_dirty=true;
//     coarse_lattice->m_hash_table->clear();
//     coarse_lattice->m_hash_table->update_impl();


//     coarse_lattice->begin_splat();
//     coarse_lattice->m_hash_table->update_impl();

//     coarse_lattice->just_create_verts(positions_raw, false);


//     return coarse_lattice;

// }



// torch::Tensor Lattice::slice_standalone_with_precomputation(torch::Tensor& positions_raw, torch::Tensor& splatting_indices_tensor, torch::Tensor& splatting_weights_tensor){

//     check_positions(positions_raw); 
//     CHECK(val_dim()>0) << "m_val_dim is 0 or lwoer. We have to splat something first so that we have values from where to slice. Val dim is " << val_dim();
//     int nr_positions=positions_raw.size(0);
//     int pos_dim=positions_raw.size(1);
//     CHECK(pos_dim==this->pos_dim()) << " The position dimension do not coreespond with the ones we used for creating the lattice";
//     splatting_indices_tensor=splatting_indices_tensor.contiguous();
//     splatting_weights_tensor=splatting_weights_tensor.contiguous();


//      //to cuda
//     TIME_START("upload_cuda");
//     positions_raw=positions_raw.to("cuda");
//     m_sigmas_tensor=m_sigmas_tensor.to("cuda");
//     TIME_END("upload_cuda");

//     TIME_START("scale_by_sigma");
//     VLOG(3) << "slice standalone scaling by a sigma of " << m_sigmas_tensor;
//     Tensor positions=positions_raw/m_sigmas_tensor;
//     TIME_END("scale_by_sigma")

//     //initialize the output values to zero 
//     Tensor sliced_values_hom_tensor=torch::zeros({nr_positions, val_dim() }, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0) );


//     //assume we have already splatting weight and indices
//     if( !splatting_indices_tensor.defined() || !splatting_weights_tensor.defined()  || splatting_indices_tensor.size(0)!=nr_positions*(this->pos_dim()+1) ||  splatting_weights_tensor.size(0)!=nr_positions*(this->pos_dim()+1)  ){
//         LOG(FATAL) << "Indices or wegiths tensor is not created or doesnt have the correct size. We are assuming it has size " << nr_positions*(this->pos_dim()+1) << "but indices has size " << splatting_indices_tensor.sizes() << " m_splatting_weights_tensor have size "  << splatting_weights_tensor.sizes();
//     }
//     m_hash_table->update_impl();




//     TIME_START("slice");
//     m_impl->slice_standalone_with_precomputation( positions.data_ptr<float>(), sliced_values_hom_tensor.data_ptr<float>(), this->pos_dim(), this->val_dim(),  nr_positions, splatting_indices_tensor.data_ptr<int>(), splatting_weights_tensor.data_ptr<float>(), *(m_hash_table->m_impl) );
//     TIME_END("slice");


//     return sliced_values_hom_tensor;


// }

// std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> Lattice::slice_standalone_no_precomputation(torch::Tensor& positions_raw){

//     check_positions(positions_raw); 
//     CHECK(val_dim()>0) << "m_val_dim is 0 or lwoer. We have to splat something first so that we have values from where to slice. Val dim is " << val_dim();
//     int nr_positions=positions_raw.size(0);
//     int pos_dim=positions_raw.size(1);
//     CHECK(pos_dim==this->pos_dim()) << " The position dimension do not coreespond with the ones we used for creating the lattice";


//      //to cuda
//     TIME_START("upload_cuda");
//     positions_raw=positions_raw.to("cuda");
//     m_sigmas_tensor=m_sigmas_tensor.to("cuda");
//     TIME_END("upload_cuda");

//     TIME_START("scale_by_sigma");
//     VLOG(3) << "slice standalone scaling by a sigma of " << m_sigmas_tensor;
//     Tensor positions=positions_raw/m_sigmas_tensor;
//     TIME_END("scale_by_sigma")

//     //initialize the output values to zero 
//     Tensor sliced_values_hom_tensor=torch::zeros({nr_positions, val_dim() }, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0) );

//     //recalculate the splatting indices and weight for the backward pass of the slice
//     Tensor splatting_indices_tensor = torch::empty({nr_positions*(pos_dim+1) }, torch::dtype(torch::kInt32).device(torch::kCUDA, 0) );
//     Tensor splatting_weights_tensor = torch::empty({nr_positions*(pos_dim+1) }, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0) );
//     splatting_indices_tensor.fill_(-1);
//     splatting_weights_tensor.fill_(-1);

//     m_hash_table->update_impl();


//     TIME_START("slice");
//     m_impl->slice_standalone_no_precomputation( positions.data_ptr<float>(), sliced_values_hom_tensor.data_ptr<float>(), this->pos_dim(), this->val_dim(),  nr_positions, splatting_indices_tensor.data_ptr<int>(), splatting_weights_tensor.data_ptr<float>(), *(m_hash_table->m_impl) );
//     TIME_END("slice");


//     auto ret = std::make_tuple (sliced_values_hom_tensor, splatting_indices_tensor, splatting_weights_tensor ); 
//     return ret;

//     // return sliced_values_hom_tensor.clone()


// }

// std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> Lattice::slice_with_collisions_standalone_no_precomputation(torch::Tensor& positions_raw){
//     check_positions(positions_raw); 
//     CHECK(val_dim()>0) << "m_val_dim is 0 or lwoer. We have to splat something first so that we have values from where to slice. Val dim is " << val_dim();
//     int nr_positions=positions_raw.size(0);
//     int pos_dim=positions_raw.size(1);
//     CHECK(pos_dim==this->pos_dim()) << " The position dimension do not coreespond with the ones we used for creating the lattice";


//      //to cuda
//     TIME_START("upload_cuda");
//     positions_raw=positions_raw.to("cuda");
//     m_sigmas_tensor=m_sigmas_tensor.to("cuda");
//     TIME_END("upload_cuda");

//     TIME_START("scale_by_sigma");
//     VLOG(3) << "slice standalone scaling by a sigma of " << m_sigmas_tensor;
//     Tensor positions=positions_raw/m_sigmas_tensor;
//     TIME_END("scale_by_sigma")

//     //initialize the output values to zero 
//     Tensor sliced_values_hom_tensor=torch::zeros({nr_positions, val_dim() }, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0) );

//     //recalculate the splatting indices and weight for the backward pass of the slice
//     Tensor splatting_indices_tensor = torch::empty({nr_positions*(pos_dim+1) }, torch::dtype(torch::kInt32).device(torch::kCUDA, 0) );
//     Tensor splatting_weights_tensor = torch::empty({nr_positions*(pos_dim+1) }, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0) );
//     splatting_indices_tensor.fill_(-1);
//     splatting_weights_tensor.fill_(-1);

//     m_hash_table->update_impl();


//     TIME_START("slice");
//     m_impl->slice_with_collisions_standalone_no_precomputation( positions.data_ptr<float>(), sliced_values_hom_tensor.data_ptr<float>(), this->pos_dim(), this->val_dim(),  nr_positions, splatting_indices_tensor.data_ptr<int>(), splatting_weights_tensor.data_ptr<float>(), *(m_hash_table->m_impl) );
//     TIME_END("slice");


//     auto ret = std::make_tuple (sliced_values_hom_tensor, splatting_indices_tensor, splatting_weights_tensor ); 
//     return ret;
// }


// std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> Lattice::gather_standalone_no_precomputation(torch::Tensor& positions_raw){

//     check_positions(positions_raw); 
//     CHECK(val_dim()>0) << "m_val_dim is 0 or lwoer. We have to splat something first so that we have values from where to slice. Val dim is " << val_dim();
//     int nr_positions=positions_raw.size(0);
//     int pos_dim=positions_raw.size(1);
//     CHECK(pos_dim==this->pos_dim()) << " The position dimension do not coreespond with the ones we used for creating the lattice";


//      //to cuda
//     TIME_START("upload_cuda");
//     positions_raw=positions_raw.to("cuda");
//     m_sigmas_tensor=m_sigmas_tensor.to("cuda");
//     TIME_END("upload_cuda");

//     TIME_START("scale_by_sigma");
//     VLOG(3) << "gather standalone scaling by a sigma of " << m_sigmas_tensor;
//     Tensor positions=positions_raw/m_sigmas_tensor;
//     TIME_END("scale_by_sigma")

//     //initialize the output values to zero 
//     int row_size_gathered=(pos_dim+1)*(val_dim()+1); //we have m_pos_dim+1 vertices in a lattice and each has values of m_val_full_dim plus a barycentric coord
//     Tensor gathered_values_tensor=torch::zeros({nr_positions, row_size_gathered}, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0) );

//     //recalculate the splatting indices and weight for the backward pass of the gather
//     Tensor splatting_indices_tensor = torch::empty({nr_positions*(pos_dim+1) }, torch::dtype(torch::kInt32).device(torch::kCUDA, 0) );
//     Tensor splatting_weights_tensor = torch::empty({nr_positions*(pos_dim+1) }, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0) );
//     splatting_indices_tensor.fill_(-1);
//     splatting_weights_tensor.fill_(-1);
//     m_hash_table->update_impl();


//     TIME_START("gather");
//     m_impl->gather_standalone_no_precomputation( positions.data_ptr<float>(), gathered_values_tensor.data_ptr<float>(), this->pos_dim(), this->val_dim(),  nr_positions, splatting_indices_tensor.data_ptr<int>(), splatting_weights_tensor.data_ptr<float>(), *(m_hash_table->m_impl) );
//     TIME_END("gather");

//     auto ret = std::make_tuple (gathered_values_tensor, splatting_indices_tensor, splatting_weights_tensor ); 
//     return ret;
//     // return gathered_values_tensor;

// }


// torch::Tensor Lattice::gather_standalone_with_precomputation(torch::Tensor& positions_raw, torch::Tensor& splatting_indices_tensor, torch::Tensor& splatting_weights_tensor){

//     check_positions(positions_raw); 
//     CHECK(val_dim()>0) << "m_val_dim is 0 or lwoer. We have to splat something first so that we have values from where to slice. Val dim is " << val_dim();
//     int nr_positions=positions_raw.size(0);
//     int pos_dim=positions_raw.size(1);
//     CHECK(pos_dim==this->pos_dim()) << " The position dimension do not coreespond with the ones we used for creating the lattice";
//     splatting_indices_tensor=splatting_indices_tensor.contiguous();
//     splatting_weights_tensor=splatting_weights_tensor.contiguous();


//      //to cuda
//     // TIME_START("upload_cuda");
//     positions_raw=positions_raw.to("cuda");
//     m_sigmas_tensor=m_sigmas_tensor.to("cuda");
//     // TIME_END("upload_cuda");

//     // TIME_START("scale_by_sigma");
//     VLOG(3) << "gather standalone scaling by a sigma of " << m_sigmas_tensor;
//     Tensor positions=positions_raw/m_sigmas_tensor;
//     // TIME_END("scale_by_sigma")

//     //initialize the output values to zero 
//     int row_size_gathered=(this->pos_dim()+1)*(this->val_dim()+1); //we have m_pos_dim+1 vertices in a lattice and each has values of m_val_full_dim plus a barycentric coord
//     Tensor gathered_values_tensor=torch::zeros({nr_positions, row_size_gathered}, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0) );

//     //assume we have already splatting weight and indices
//     if( !splatting_indices_tensor.defined() || !splatting_weights_tensor.defined()  || splatting_indices_tensor.size(0)!=nr_positions*(this->pos_dim()+1) ||  splatting_weights_tensor.size(0)!=nr_positions*(this->pos_dim()+1)  ){
//         LOG(FATAL) << "Indices or wegiths tensor is not created or doesnt have the correct size. We are assuming it has size " << nr_positions*(this->pos_dim()+1) << "but indices has size " << splatting_indices_tensor.sizes() << " m_splatting_weights_tensor have size "  << splatting_weights_tensor.sizes();
//     }
//     m_hash_table->update_impl();


//     // TIME_START("gather");
//     m_impl->gather_standalone_with_precomputation( positions.data_ptr<float>(), gathered_values_tensor.data_ptr<float>(), this->pos_dim(), this->val_dim(),  nr_positions, splatting_indices_tensor.data_ptr<int>(), splatting_weights_tensor.data_ptr<float>(), *(m_hash_table->m_impl) );
//     // TIME_END("gather");

//     return gathered_values_tensor;

// }


// std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>  Lattice::slice_classify_no_precomputation(torch::Tensor& positions_raw, torch::Tensor& delta_weights, torch::Tensor& linear_clasify_weight, torch::Tensor& linear_clasify_bias, const int nr_classes){


//     check_positions(positions_raw); 
//     CHECK(val_dim()>0) << "m_val_dim is 0 or lwoer. We have to splat something first so that we have values from where to slice. Val dim is " << val_dim();
//     int nr_positions=positions_raw.size(0);
//     int pos_dim=positions_raw.size(1);
//     CHECK(pos_dim==this->pos_dim()) << " The position dimension do not coreespond with the ones we used for creating the lattice";
//     delta_weights=delta_weights.contiguous();
//     linear_clasify_weight=linear_clasify_weight.contiguous();
//     linear_clasify_bias=linear_clasify_bias.contiguous();



//      //to cuda
//     TIME_START("upload_cuda");
//     positions_raw=positions_raw.to("cuda");
//     m_sigmas_tensor=m_sigmas_tensor.to("cuda");
//     delta_weights=delta_weights.to("cuda");
//     linear_clasify_weight=linear_clasify_weight.to("cuda");
//     linear_clasify_bias=linear_clasify_bias.to("cuda");
//     TIME_END("upload_cuda");

//     TIME_START("scale_by_sigma");
//     VLOG(3) << "slice standalone scaling by a sigma of " << m_sigmas_tensor;
//     Tensor positions=positions_raw/m_sigmas_tensor;
//     TIME_END("scale_by_sigma")

//     //we store here the class logits directly
//     Tensor sliced_values_hom_tensor=torch::zeros({nr_positions, nr_classes}, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0) );


//     //recalculate the splatting indices and weight for the backward pass of the slice
//     Tensor splatting_indices_tensor = torch::empty({nr_positions*(pos_dim+1) }, torch::dtype(torch::kInt32).device(torch::kCUDA, 0) );
//     Tensor splatting_weights_tensor = torch::empty({nr_positions*(pos_dim+1) }, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0) );
//     splatting_indices_tensor.fill_(-1);
//     splatting_weights_tensor.fill_(-1);
//     m_hash_table->update_impl();


//     TIME_START("slice_classify");
//     m_impl->slice_classify_no_precomputation( positions.data_ptr<float>(), 
//                                               sliced_values_hom_tensor.data_ptr<float>(), 
//                                               delta_weights.data_ptr<float>(), 
//                                               linear_clasify_weight.data_ptr<float>(), 
//                                               linear_clasify_bias.data_ptr<float>(), 
//                                               nr_classes,
//                                               this->pos_dim(), 
//                                               this->val_dim(),  
//                                               nr_positions, 
//                                               splatting_indices_tensor.data_ptr<int>(), 
//                                               splatting_weights_tensor.data_ptr<float>(), 
//                                               *(m_hash_table->m_impl) );
//     TIME_END("slice_classify");

//     auto ret = std::make_tuple (sliced_values_hom_tensor, splatting_indices_tensor, splatting_weights_tensor ); 
//     return ret;
//     // return sliced_values_hom_tensor;

// }


// torch::Tensor Lattice::slice_classify_with_precomputation(torch::Tensor& positions_raw, torch::Tensor& delta_weights, torch::Tensor& linear_clasify_weight, torch::Tensor& linear_clasify_bias, const int nr_classes, torch::Tensor& splatting_indices_tensor, torch::Tensor& splatting_weights_tensor){

//     check_positions(positions_raw); 
//     CHECK(val_dim()>0) << "m_val_dim is 0 or lwoer. We have to splat something first so that we have values from where to slice. Val dim is " << val_dim();
//     int nr_positions=positions_raw.size(0);
//     int pos_dim=positions_raw.size(1);
//     CHECK(pos_dim==this->pos_dim()) << " The position dimension do not coreespond with the ones we used for creating the lattice";
//     delta_weights=delta_weights.contiguous();
//     linear_clasify_weight=linear_clasify_weight.contiguous();
//     linear_clasify_bias=linear_clasify_bias.contiguous();
//     splatting_indices_tensor=splatting_indices_tensor.contiguous();
//     splatting_weights_tensor=splatting_weights_tensor.contiguous();



//      //to cuda
//     // TIME_START("upload_cuda");
//     positions_raw=positions_raw.to("cuda");
//     m_sigmas_tensor=m_sigmas_tensor.to("cuda");
//     delta_weights=delta_weights.to("cuda");
//     linear_clasify_weight=linear_clasify_weight.to("cuda");
//     linear_clasify_bias=linear_clasify_bias.to("cuda");
//     // TIME_END("upload_cuda");

//     // TIME_START("scale_by_sigma");
//     VLOG(3) << "slice standalone scaling by a sigma of " << m_sigmas_tensor;
//     Tensor positions=positions_raw/m_sigmas_tensor;
//     // TIME_END("scale_by_sigma")

//     //we store here the class logits directly
//     Tensor sliced_values_hom_tensor=torch::zeros({nr_positions, nr_classes}, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0) );


//     //assume we have already splatting weight and indices
//     if( !splatting_indices_tensor.defined() || !splatting_weights_tensor.defined()  || splatting_indices_tensor.size(0)!=nr_positions*(this->pos_dim()+1) ||  splatting_weights_tensor.size(0)!=nr_positions*(this->pos_dim()+1)  ){
//         LOG(FATAL) << "Indices or wegiths tensor is not created or doesnt have the correct size. We are assuming it has size " << nr_positions*(this->pos_dim()+1) << "but indices has size " << splatting_indices_tensor.sizes() << " m_splatting_weights_tensor have size "  << splatting_weights_tensor.sizes();
//     }
//     m_hash_table->update_impl();


//     // TIME_START("slice_classify_cuda");
//     m_impl->slice_classify_with_precomputation( positions.data_ptr<float>(), 
//                                               sliced_values_hom_tensor.data_ptr<float>(), 
//                                               delta_weights.data_ptr<float>(), 
//                                               linear_clasify_weight.data_ptr<float>(), 
//                                               linear_clasify_bias.data_ptr<float>(), 
//                                               nr_classes,
//                                               this->pos_dim(), 
//                                               this->val_dim(),  
//                                               nr_positions, 
//                                               splatting_indices_tensor.data_ptr<int>(), 
//                                               splatting_weights_tensor.data_ptr<float>(), 
//                                               *(m_hash_table->m_impl) );
//     // TIME_END("slice_classify_cuda");

//     return sliced_values_hom_tensor;

// }





// void Lattice::slice_backwards_standalone_with_precomputation(torch::Tensor& positions_raw, const torch::Tensor& sliced_values_hom, const Tensor& grad_sliced_values, torch::Tensor& splatting_indices_tensor, torch::Tensor& splatting_weights_tensor){

//     check_positions(positions_raw); 
//     CHECK(val_dim()>0) << "m_val_dim is 0 or lwoer. We have to splat something first so that we have values from where to slice. Val dim is " << val_dim();
//     int nr_positions=positions_raw.size(0);
//     int pos_dim=positions_raw.size(1);
//     CHECK(pos_dim==this->pos_dim()) << " The position dimension do not coreespond with the ones we used for creating the lattice";
//     CHECK(grad_sliced_values.is_contiguous()) << "Grad sliced values needs to be contiguous. Please call .contiguous() on it";
//     CHECK(sliced_values_hom.is_contiguous()) << "sliced_values_hom needs to be contiguous. Please call .contiguous() on it";
//     splatting_indices_tensor=splatting_indices_tensor.contiguous();
//     splatting_weights_tensor=splatting_weights_tensor.contiguous();




//     TIME_START("slice_back");
//     m_impl->slice_backwards_standalone_with_precomputation( sliced_values_hom.data_ptr<float>(), grad_sliced_values.data_ptr<float>(), splatting_indices_tensor.data_ptr<int>(), splatting_weights_tensor.data_ptr<float>(), this->pos_dim(), this->val_dim(), nr_positions, *(m_hash_table->m_impl) );
//     TIME_END("slice_back");

// }


// void Lattice::slice_backwards_standalone_with_precomputation_no_homogeneous(torch::Tensor& positions_raw, const Tensor& grad_sliced_values, torch::Tensor& splatting_indices_tensor, torch::Tensor& splatting_weights_tensor){

//     check_positions(positions_raw); 
//     CHECK(val_dim()>0) << "m_val_dim is 0 or lwoer. We have to splat something first so that we have values from where to slice. Val dim is " << val_dim();
//     int nr_positions=positions_raw.size(0);
//     int pos_dim=positions_raw.size(1);
//     CHECK(pos_dim==this->pos_dim()) << " The position dimension do not coreespond with the ones we used for creating the lattice";
//     CHECK(grad_sliced_values.dim()==2) <<"grad_sliced_values should be nr_positions x m_val_dim, so it should have 2 dimensions. However it has "<< grad_sliced_values.dim();
//     CHECK(grad_sliced_values.is_contiguous()) << "Grad sliced values needs to be contiguous. Please call .contiguous() on it";
//     splatting_indices_tensor=splatting_indices_tensor.contiguous();
//     splatting_weights_tensor=splatting_weights_tensor.contiguous();

//     m_hash_table->m_values_tensor=torch::zeros({nr_lattice_vertices(), grad_sliced_values.size(1)},  torch::dtype(torch::kFloat32).device(torch::kCUDA, 0)  );
//     m_hash_table->update_impl();



//     TIME_START("slice_back");
//     m_impl->slice_backwards_standalone_with_precomputation_no_homogeneous(grad_sliced_values.data_ptr<float>(), splatting_indices_tensor.data_ptr<int>(), splatting_weights_tensor.data_ptr<float>(), this->pos_dim(), this->val_dim(), nr_positions, *(m_hash_table->m_impl) );
//     TIME_END("slice_back");

// }


// void Lattice::slice_classify_backwards_with_precomputation(const torch::Tensor& grad_class_logits, torch::Tensor& positions_raw, torch::Tensor& initial_values, torch::Tensor& delta_weights, torch::Tensor&  linear_clasify_weight, torch::Tensor& linear_clasify_bias, const int nr_classes, torch::Tensor& grad_lattice_values, torch::Tensor& grad_delta_weights, torch::Tensor& grad_linear_clasify_weight, torch::Tensor& grad_linear_clasify_bias, torch::Tensor& splatting_indices_tensor, torch::Tensor& splatting_weights_tensor){

//     check_positions(positions_raw); 
//     CHECK(val_dim()>0) << "m_val_dim is 0 or lwoer. We have to splat something first so that we have values from where to slice. Val dim is " << val_dim();
//     int nr_positions=positions_raw.size(0);
//     int pos_dim=positions_raw.size(1);
//     CHECK(pos_dim==this->pos_dim()) << " The position dimension do not coreespond with the ones we used for creating the lattice";
//     CHECK(grad_class_logits.dim()==2) <<"grad_class_logits should be  nr_positions x nr_classes, so it should have 2 dimensions. However it has "<< grad_class_logits.dim();
//     // m_val_dim=initial_values.size(1);
//     CHECK(grad_class_logits.is_contiguous()) << "grad_class_logits needs to be contiguous. Please call .contiguous() on it";
//     initial_values=initial_values.contiguous();
//     delta_weights=delta_weights.contiguous();
//     linear_clasify_weight=linear_clasify_weight.contiguous();
//     linear_clasify_bias=linear_clasify_bias.contiguous();
//     splatting_indices_tensor=splatting_indices_tensor.contiguous();
//     splatting_weights_tensor=splatting_weights_tensor.contiguous();


//     // TIME_START("slice_clasify_back");
//     m_impl->slice_classify_backwards_with_precomputation(grad_class_logits.data_ptr<float>(), initial_values.data_ptr<float>(),  splatting_indices_tensor.data_ptr<int>(), splatting_weights_tensor.data_ptr<float>(), this->pos_dim(), this->val_dim(), nr_positions,
//     delta_weights.data_ptr<float>(), linear_clasify_weight.data_ptr<float>(), linear_clasify_bias.data_ptr<float>(), nr_classes, grad_lattice_values.data_ptr<float>(), grad_delta_weights.data_ptr<float>(), grad_linear_clasify_weight.data_ptr<float>(),grad_linear_clasify_bias.data_ptr<float>(),
//      *(m_hash_table->m_impl) );
//     // TIME_END("slice_clasify_back");

// }

// void Lattice::gather_backwards_standalone_with_precomputation(const torch::Tensor& positions_raw, const Tensor& grad_sliced_values, torch::Tensor& splatting_indices_tensor, torch::Tensor& splatting_weights_tensor){

//     check_positions(positions_raw); 
//     CHECK(val_dim()>0) << "m_val_dim is 0 or lwoer. We have to splat something first so that we have values from where to slice. Val dim is " << val_dim();
//     int nr_positions=positions_raw.size(0);
//     int pos_dim=positions_raw.size(1);
//     int val_dim=grad_sliced_values.size(1)/(pos_dim+1)-1; //we will acumulate the gradient into the value tensor. And it should have the same val_dim as the values that were in the lattice_we_gathered from
//     CHECK(pos_dim==this->pos_dim()) << " The position dimension do not coreespond with the ones we used for creating the lattice";
//     CHECK(grad_sliced_values.dim()==2) <<"grad_sliced_values should be nr_positions x ((m_val_dim+1)*(m_pos_dim+1)), so it should have 2 dimensions. However it has "<< grad_sliced_values.dim();
//     CHECK(grad_sliced_values.is_contiguous()) << "Grad sliced values needs to be contiguous. Please call .contiguous() on it";
//     splatting_indices_tensor=splatting_indices_tensor.contiguous();
//     splatting_weights_tensor=splatting_weights_tensor.contiguous();



//     m_hash_table->m_values_tensor=torch::zeros({nr_lattice_vertices(), val_dim },  torch::dtype(torch::kFloat32).device(torch::kCUDA, 0)  );
//     m_hash_table->update_impl();



//     // TIME_START("gather_back");
//     m_impl->gather_backwards_standalone_with_precomputation(grad_sliced_values.data_ptr<float>(), splatting_indices_tensor.data_ptr<int>(), splatting_weights_tensor.data_ptr<float>(), this->pos_dim(),  this->val_dim(), nr_positions, *(m_hash_table->m_impl) );
//     // TIME_END("gather_back");


// }



// std::shared_ptr<Lattice> Lattice::clone_lattice(){
//     std::shared_ptr<Lattice> new_lattice=create(this); //create a lattice with no config but takes the config from this one
//     return new_lattice;
// }



// void Lattice::increase_sigmas(const float stepsize){
//         // m_sigmas.clear();
//     for(size_t i=0; i<m_sigmas.size(); i++){
//         m_sigmas[i]+=stepsize;
//     }

//     m_sigmas_tensor=vec2tensor(m_sigmas);

// }








//getters
// int Lattice::val_dim(){
    // return m_hash_table->val_dim();
// }
int Lattice::pos_dim(){
    // return m_hash_table->pos_dim();
    return m_expected_pos_dim;
}
int Lattice::capacity(){
    // return m_hash_table->capacity();
    return m_capacity;
}
// std::string Lattice::name(){
//     return m_name;
// }
// int Lattice::nr_lattice_vertices(){
  
//     // m_impl->wait_to_create_vertices(); //we synchronize the event and wait until whatever kernel was launched to create vertices has also finished
//     // // cudaEventSynchronize(m_event_nr_vertices_lattice_changed);  //we synchronize the event and wait until whatever kernel was launched to create vertices has also finished
//     // int nr_verts=0;
//     // cudaMemcpy ( &nr_verts,  m_hash_table->m_nr_filled_tensor.data_ptr<int>(), sizeof(int), cudaMemcpyDeviceToHost );
//     // CHECK(nr_verts>=0) << "nr vertices cannot be negative. However it is " << nr_verts;
//     // CHECK(nr_verts<1e+8) << "nr vertices cannot be that high. However it is " << nr_verts;
//     // return nr_verts;


//     //attempt 2  
//     //check if the nr_latttice_vertices is dirty which means that a kernel has been executed that might have modified the nr of vertices
//     int nr_verts=0;
//     if (m_hash_table->m_nr_filled_is_dirty){
//         m_hash_table->m_nr_filled_is_dirty=false;
//         cudaMemcpy ( &nr_verts,  m_hash_table->m_nr_filled_tensor.data_ptr<int>(), sizeof(int), cudaMemcpyDeviceToHost );
//         m_hash_table->m_nr_filled=nr_verts;
//     }else{
//         // return number lattice vertices that the cpu knows about
//         nr_verts=m_hash_table->m_nr_filled;

//         // //sanity check if the value is the same as we read it again
//         // int check_nr_verts;
//         // cudaMemcpy ( &check_nr_verts,  m_hash_table->m_nr_filled_tensor.data_ptr<int>(), sizeof(int), cudaMemcpyDeviceToHost );
//         // CHECK(check_nr_verts==nr_verts) << "Value is not the same. Checked nr of verts is " << check_nr_verts << " nr verts that we wanted to return is " << nr_verts;
//     }

//     CHECK(nr_verts>=0) << "nr vertices cannot be negative. However it is " << nr_verts;
//     CHECK(nr_verts<1e+8) << "nr vertices cannot be that high. However it is " << nr_verts;

//     return nr_verts;
// }
// int Lattice::get_filter_extent(const int neighborhood_size) {
//     CHECK(neighborhood_size==1) << "At the moment we only have implemented a filter with a neighbourhood size of 1. I haven't yet written the more general formula for more neighbourshood size";
//     CHECK(this->pos_dim()!=-1) << "m pos dim is not set. It is -1";

//     return 2*(this->pos_dim()+1) + 1; //because we have 2 neighbour for each axis and we have pos_dim+1 axes. Also a +1 for the center vertex
// }

// torch::Tensor Lattice::compute_scale_factor_tensor(const std::vector<float> sigmas_list, const int pos_dim){

//     int nr_resolutions=sigmas_list.size();

//     Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> scale_factor_eigen;
//     scale_factor_eigen.resize(nr_resolutions, pos_dim);
//     double invStdDev = 1.0;
//     for(int res_idx=0; res_idx<nr_resolutions; res_idx++){
//         for (int i = 0; i < pos_dim; i++) {
//             scale_factor_eigen(res_idx,i) =  1.0 / (std::sqrt((double) (i + 1) * (i + 2))) * invStdDev;
//             scale_factor_eigen(res_idx,i)=scale_factor_eigen(res_idx,i)/ sigmas_list[res_idx];
//             // VLOG(1) << "scalinbg by " << sigmas_list[res_idx];
//         }
//     }
//     // VLOG(1) << "scale_factor_eigen" << scale_factor_eigen;
//     Tensor scale_factor_tensor=eigen2tensor(scale_factor_eigen.cast<float>()).cuda();
//     scale_factor_tensor=scale_factor_tensor.view({nr_resolutions, pos_dim}); //nr_resolutuons x pos_dim

//     return scale_factor_tensor;

// }

torch::Tensor Lattice::sigmas_tensor(){
    return m_sigmas_tensor;
}
// torch::Tensor Lattice::positions(){
    // return m_positions;
// }
// std::shared_ptr<HashTable> Lattice::hash_table(){
//     return m_hash_table;
// }
// torch::Tensor Lattice::values(){
//     return  m_hash_table->m_values_tensor;
// }
bool Lattice::is_half_precision(){
    #if LATTICE_HALF_PRECISION
        return true;
    #else
        return false;
    #endif
}



//setters
// void Lattice::set_sigma(const float sigma){
//     int nr_sigmas=m_sigmas_val_and_extent.size();
//     CHECK(nr_sigmas==1) << "We are summing we have onyl one sigma. This method is intended to affect only one and not two sigmas independently";

//     for(size_t i=0; i<m_sigmas.size(); i++){
//         m_sigmas[i]=sigma;
//     }

//     m_sigmas_tensor=vec2tensor(m_sigmas);
// }
// void Lattice::set_name(const std::string name){
//     m_name=name;
// }
// void Lattice::set_values(const torch::Tensor& new_values, const bool sanity_check){
//     // m_values_tensor=new_values.contiguous();
//     // update_impl();
//     m_hash_table->set_values(new_values);
//     if (sanity_check){
//         CHECK(new_values.size(0)==nr_lattice_vertices()) << "The nr of rows in the new values does not correspond to the nr_lattice_vertices. Nr of rows is " << new_values.size(0) << " and nr lattice vertices is " << nr_lattice_vertices();
//     }
// }
// void Lattice::set_positions( const torch::Tensor& positions_raw ){
//     m_positions=positions_raw;
// }
// void Lattice::set_nr_lattice_vertices(const int nr_verts){
//     m_hash_table->m_nr_filled_is_dirty=false;
//     m_hash_table->m_nr_filled=nr_verts;
//     cudaMemcpy (  m_hash_table->m_nr_filled_tensor.data_ptr<int>(),  &nr_verts, sizeof(int), cudaMemcpyHostToDevice );
// }
 


