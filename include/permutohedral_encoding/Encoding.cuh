#pragma once

#include <memory>
#include <stdarg.h>

#include <cuda.h>


#include "torch/torch.h"


//minimum input required for any call to the encoding function. Makes it easies to declare the functions since they will all at least get this as input
struct EncodingInput{
    EncodingInput(const torch::Tensor& lattice_values, const torch::Tensor& positions_raw, const torch::Tensor& anneal_window, const bool require_lattice_values_grad, const bool require_positions_grad):
        m_lattice_values(lattice_values),
        m_positions_raw(positions_raw),
        m_anneal_window(anneal_window),
        m_require_lattice_values_grad(require_lattice_values_grad),
        m_require_positions_grad(require_positions_grad)
        {

    };
    torch::Tensor m_lattice_values;
    torch::Tensor m_positions_raw;
    torch::Tensor m_anneal_window;
    bool m_require_lattice_values_grad;
    bool m_require_positions_grad;    
};
//fixed params that do not change during the lifetime of the lattice
struct EncodingFixedParams{
    EncodingFixedParams(const int pos_dim, const int capacity, const int nr_levels, const int nr_feat_per_level, const std::vector<float>& sigmas_list, const torch::Tensor& random_shift_per_level, const bool concat_points, const float points_scaling):
        m_pos_dim(pos_dim),
        m_capacity(capacity),
        m_nr_levels(nr_levels),
        m_nr_feat_per_level(nr_feat_per_level),
        m_sigmas_list(sigmas_list),
        m_scale_factor(compute_scale_factor_tensor(sigmas_list, pos_dim)),
        m_random_shift_per_level(random_shift_per_level),
        m_concat_points(concat_points),
        m_points_scaling(points_scaling)
        {

    };
    const int m_pos_dim;
    const int m_capacity;
    const int m_nr_levels;
    const int m_nr_feat_per_level;
    std::vector<float> m_sigmas_list;
    torch::Tensor m_scale_factor;
    torch::Tensor m_random_shift_per_level;
    bool m_concat_points;
    float m_points_scaling;

    torch::Tensor compute_scale_factor_tensor(const std::vector<float> sigmas_list, const int pos_dim){
        int nr_resolutions=sigmas_list.size();

        torch::Tensor scale_factor_tensor=torch::zeros({ nr_resolutions, pos_dim },  torch::dtype(torch::kFloat32).device(torch::kCUDA, 0)  );
        double invStdDev = 1.0;
        for(int res_idx=0; res_idx<nr_resolutions; res_idx++){
            for (int i = 0; i < pos_dim; i++) {
                scale_factor_tensor[res_idx][i] =  1.0 / (std::sqrt((double) (i + 1) * (i + 2))) * invStdDev;
                scale_factor_tensor[res_idx][i]=scale_factor_tensor[res_idx][i]/ sigmas_list[res_idx];
            }
        }

        return scale_factor_tensor;

    }
};






// In order to create an encoding we need to return a templated version of it with the create_encoding factory.
// we create a baseclass with no templates so we can return that one 
//https://stackoverflow.com/questions/71852071/how-to-create-a-factory-function-for-a-templated-class
class EncodingBase{
public:
    // EncodingBase();
    // ~EncodingBase();    
    // virtual std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward(std::forward<Args>(args)...);
    // virtual std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward(const torch::Tensor& lattice_values, const torch::Tensor& scale_factor, torch::Tensor& positions_raw, torch::Tensor& random_shift_per_level, torch::Tensor& anneal_window, const bool concat_points, const float points_scaling, const bool require_lattice_values_grad, const bool require_positions_grad);###works
    virtual std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward(const EncodingInput& input) =0;
    virtual std::tuple<torch::Tensor, torch::Tensor> backward(const EncodingInput& input, torch::Tensor& grad_sliced_values_monolithic) =0;
    virtual std::tuple<torch::Tensor, torch::Tensor> double_backward(const EncodingInput& input, const torch::Tensor& double_positions_grad, torch::Tensor& grad_sliced_values_monolithic) =0;
        
};





template<uint32_t POS_DIM, uint32_t NR_FEAT_PER_LEVEL>
class Encoding : public EncodingBase, public std::enable_shared_from_this<Encoding<POS_DIM,NR_FEAT_PER_LEVEL>>{
public:
    template <class ...Args>
    static std::shared_ptr<Encoding> create( Args&& ...args ){
        return std::shared_ptr<Encoding>( new Encoding(std::forward<Args>(args)...) );
    }
    ~Encoding();
    Encoding(const EncodingFixedParams& fixed_params);

    // void test(const torch::Tensor& tensor);


    //forward pass, does a slice from the lattice
    // std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward(const torch::Tensor& lattice_values, const torch::Tensor& scale_factor, torch::Tensor& positions_raw, torch::Tensor& random_shift_per_level, torch::Tensor& anneal_window, const bool concat_points, const float points_scaling, const bool require_lattice_values_grad, const bool require_positions_grad) override;
    // std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward(const torch::Tensor& lattice_values, torch::Tensor& positions_raw, torch::Tensor& anneal_window, const bool require_lattice_values_grad, const bool require_positions_grad) override;
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward(const EncodingInput& input) override;


    //backwards passes 
    // torch::Tensor slice_backwards_standalone_with_precomputation_no_homogeneous_mr_monolithic(torch::Tensor& positions_raw, const int capacity, const torch::Tensor& grad_sliced_values, torch::Tensor& splatting_indices_tensor, torch::Tensor& splatting_weights_tensor);
    // std::tuple<torch::Tensor, torch::Tensor> slice_backwards_standalone_no_precomputation_no_homogeneous_mr_monolithic(torch::Tensor& positions_raw, torch::Tensor& lattice_values_monolithic, torch::Tensor& grad_sliced_values_monolithic, const torch::Tensor& scale_factor, torch::Tensor& random_shift_monolithic, torch::Tensor& anneal_window, const bool concat_points, const bool require_lattice_values_grad, const bool require_positions_grad);
    std::tuple<torch::Tensor, torch::Tensor> backward(const EncodingInput& input, torch::Tensor& grad_sliced_values_monolithic) override;

    //double backward
    // std::tuple<torch::Tensor, torch::Tensor> slice_double_back_from_positions_grad(const torch::Tensor& double_positions_grad, torch::Tensor& positions_raw, torch::Tensor& lattice_values_monolithic, torch::Tensor& grad_sliced_values_monolithic, const torch::Tensor& scale_factor, torch::Tensor& random_shift_monolithic, torch::Tensor& anneal_window, const bool concat_points);
    std::tuple<torch::Tensor, torch::Tensor> double_backward(const EncodingInput& input, const torch::Tensor& double_positions_grad, torch::Tensor& grad_sliced_values_monolithic);
    






   
    // torch::Tensor compute_scale_factor_tensor(const std::vector<float> sigmas_list, const int pos_dim);


    
   

   

private:
    // Encoding(const int capacity, const int pos_dim, const int nr_levels, const int nr_feat_per_level);
    void check_positions(const torch::Tensor& positions_raw);
    // void check_positions_elevated(const torch::Tensor& positions_elevated);
    void check_values(const torch::Tensor& values);
    void check_positions_and_values(const torch::Tensor& positions_raw, const torch::Tensor& values);
    // void update_impl();

    // std::string m_name;
    // int m_lvl; //lvl of coarsenes of the lattice, it starts at 1 for the finest lattice and increases by 1 for each applicaiton for coarsen()

    // std::shared_ptr<HashTable> m_hash_table;
    // std::shared_ptr<LatticeGPU> m_impl;
    // torch::Tensor m_positions; //positions that were used originally for creating the lattice
    // static int m_expected_position_dimensions;
    // int m_expected_pos_dim;
    // int m_capacity;
    // int m_nr_levels;
    // int m_nr_feat_per_level;

   
    // std::vector<float> m_sigmas;
    // torch::Tensor m_sigmas_tensor;
    // std::vector< std::pair<float, int> > m_sigmas_val_and_extent; //for each sigma we store here the value and the number of dimensions it affect. In the Gui we modify this one

    //parameters that are kept fixed and don't change from forward pass to forward pass 
    // torch::Tensor m_scale_factor;
    // torch::Tensor m_random_shift_per_level;
    // bool m_concat_points;
    // float m_points_scaling;
    EncodingFixedParams m_fixed_params;

  
};







//NOTE: more specializations can be added by uncommenting the corresponding line here AND adding the specialization at Encoding.cu
//this functions templates on pos_dim and call the nr_feat_one
template <uint32_t POS_DIM>
inline std::shared_ptr<EncodingBase> create_encoding_template_pos_dim(const int pos_dim, const int nr_feat_per_level, const EncodingFixedParams& fixed_params){
    switch (nr_feat_per_level){
        case(2): return std::make_shared<Encoding<POS_DIM, 2> >(fixed_params);
        // case(4): return std::make_shared<Encoding<POS_DIM, 4> >();
        // case(8): return std::make_shared<Encoding<POS_DIM, 8> >();
        // case(7) return std::make_shared<Encoding<POS_DIM, 2> >();
        default: throw std::runtime_error{"Encoding: nr_feat_per_level must be 2,4 or 8"};
    }  
}
//since we need to call template cuda functions we need to create a templated encoding object that has a compile time known pos dim and nr_feat_per_level
inline std::shared_ptr<EncodingBase> create_encoding(const int pos_dim, const int nr_feat_per_level, const EncodingFixedParams& fixed_params){
    switch (pos_dim){
        // case(2): return create_encoding_template_pos_dim<2>(pos_dim, nr_feat_per_level);
        case(3): return create_encoding_template_pos_dim<3>(pos_dim, nr_feat_per_level, fixed_params);
        // case(4): return create_encoding_template_pos_dim<4>(pos_dim, nr_feat_per_level);
        // case(5): return create_encoding_template_pos_dim<5>(pos_dim, nr_feat_per_level);
        // case(6): return create_encoding_template_pos_dim<6>(pos_dim, nr_feat_per_level);
        // case(7) return (2, nr_feat_per_level);
        default: throw std::runtime_error{"Encoding: pos_dim must be 2,3,4,5 or 6"};
    }
}



//In order to expose the Encoding to Pybind we would need an explicit specialization for every encoding template. Instead we expose this object and internally here we create all the encoding and template stuff and expose this   
class EncodingWrapper: public std::enable_shared_from_this<EncodingWrapper>{
public:
    template <class ...Args>
    static std::shared_ptr<EncodingWrapper> create( Args&& ...args ){
        return std::shared_ptr<EncodingWrapper>( new EncodingWrapper(std::forward<Args>(args)...) );
    }
    ~EncodingWrapper(){};

    // std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward(const torch::Tensor& lattice_values, const torch::Tensor& scale_factor, torch::Tensor& positions_raw, torch::Tensor& random_shift_per_level, torch::Tensor& anneal_window, const bool concat_points, const float points_scaling, const bool require_lattice_values_grad, const bool require_positions_grad){
    //     return m_encoding->forward(lattice_values, scale_factor, positions_raw, random_shift_per_level, anneal_window, concat_points, points_scaling, require_lattice_values_grad, require_positions_grad);
    // }//also works


    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward(const EncodingInput& input){
        return m_encoding->forward(input);
    } 
    std::tuple<torch::Tensor, torch::Tensor> backward(const EncodingInput& input, torch::Tensor& grad_sliced_values_monolithic){
        return m_encoding->backward(input, grad_sliced_values_monolithic);
    }
    std::tuple<torch::Tensor, torch::Tensor> double_backward(const EncodingInput& input, const torch::Tensor& double_positions_grad, torch::Tensor& grad_sliced_values_monolithic){
        return m_encoding->double_backward(input, double_positions_grad, grad_sliced_values_monolithic);
    }   



    // template<typename... Args> void forward(Args... args){
        // return m_encoding->forward(args...);
    // }
    // template<typename... Args> void backward(Args... args);
    // template<typename... Args> void double_backward(Args... args);


private:
    EncodingWrapper(const int pos_dim, const int nr_feat_per_level, const EncodingFixedParams& fixed_params){
        m_encoding=create_encoding(pos_dim, nr_feat_per_level, fixed_params);
    }
    std::shared_ptr<EncodingBase> m_encoding;    
};



// #include "../src/Encoding.cu"