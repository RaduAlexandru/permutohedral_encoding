#pragma once

#include <memory>
#include <stdarg.h>

#include <cuda.h>


#include "torch/torch.h"


class Encoding : public std::enable_shared_from_this<Encoding>{
public:
    template <class ...Args>
    static std::shared_ptr<Encoding> create( Args&& ...args ){
        return std::shared_ptr<Encoding>( new Encoding(std::forward<Args>(args)...) );
    }
    ~Encoding();


    static void test(const torch::Tensor& tensor);


    //forward passes
    static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> slice_with_collisions_standalone_no_precomputation_fast_mr_monolithic(const torch::Tensor& lattice_values_monolithic, const torch::Tensor& scale_factor, torch::Tensor& positions_raw, torch::Tensor& random_shift_monolithic, torch::Tensor& anneal_window, const bool concat_points, const float points_scaling, const bool require_lattice_values_grad, const bool require_positions_grad);


    //backwards passes 
    // torch::Tensor slice_backwards_standalone_with_precomputation_no_homogeneous_mr_monolithic(torch::Tensor& positions_raw, const int capacity, const torch::Tensor& grad_sliced_values, torch::Tensor& splatting_indices_tensor, torch::Tensor& splatting_weights_tensor);
    static std::tuple<torch::Tensor, torch::Tensor> slice_backwards_standalone_no_precomputation_no_homogeneous_mr_monolithic(torch::Tensor& positions_raw, torch::Tensor& lattice_values_monolithic, torch::Tensor& grad_sliced_values_monolithic, const torch::Tensor& scale_factor, torch::Tensor& random_shift_monolithic, torch::Tensor& anneal_window, const bool concat_points, const bool require_lattice_values_grad, const bool require_positions_grad);

    //double backward
    static std::tuple<torch::Tensor, torch::Tensor> slice_double_back_from_positions_grad(const torch::Tensor& double_positions_grad, torch::Tensor& positions_raw, torch::Tensor& lattice_values_monolithic, torch::Tensor& grad_sliced_values_monolithic, const torch::Tensor& scale_factor, torch::Tensor& random_shift_monolithic, torch::Tensor& anneal_window, const bool concat_points);






   
    static torch::Tensor compute_scale_factor_tensor(const std::vector<float> sigmas_list, const int pos_dim);


    
   

   

private:
    // Encoding(const int capacity, const int pos_dim, const int nr_levels, const int nr_feat_per_level);
    Encoding();
    static void check_positions(const torch::Tensor& positions_raw);
    // void check_positions_elevated(const torch::Tensor& positions_elevated);
    static void check_values(const torch::Tensor& values);
    static void check_positions_and_values(const torch::Tensor& positions_raw, const torch::Tensor& values);
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



  
};


