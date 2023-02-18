#pragma once

#include <memory>
#include <stdarg.h>

#include <cuda.h>


#include "torch/torch.h"

// #include "instant_ngp_2/jitify_helper/jitify_options.hpp" //Needs to be added BEFORE jitify because this defined the include paths so that the kernels cna find each other
// #include "jitify/jitify.hpp"
// #include <Eigen/Dense>


class LatticeGPU;
// class HashTable;

// class Lattice : public torch::autograd::Variable, public std::enable_shared_from_this<Lattice>{
// class Lattice : public at::Tensor, public std::enable_shared_from_this<Lattice>{
class Lattice : public std::enable_shared_from_this<Lattice>{
// class Lattice : public torch::Tensor, public std::enable_shared_from_this<Lattice>{
// class Lattice :public THPVariable, public std::enable_shared_from_this<Lattice>{
public:
    template <class ...Args>
    static std::shared_ptr<Lattice> create( Args&& ...args ){
        return std::shared_ptr<Lattice>( new Lattice(std::forward<Args>(args)...) );
    }
    ~Lattice();

    void init(const int val_dim);
    void clear();
    void clear_only_values();

    void set_sigmas_from_string(std::string sigma_val_and_extent); //we can set only one sigma from here but that should be enough for most purposes
    void set_sigmas(std::initializer_list<  std::pair<float, int> > sigmas_list); //its nice to call as a whole function which gets a list of std pairs
    void set_sigmas(std::vector<  std::pair<float, int> > sigmas_list); // in the init_params code I need to pass an explicit std vector so in this case I would need this
    // torch::Tensor bilateral_filter(torch::Tensor& positions_raw, torch::Tensor& values); //runs a bilateral filter on the positions and values and returns the output values

    // void begin_splat(const bool reset_hashmap=true); //clears the hashtable and new_values matris so we can use them as fresh
    // std::tuple<torch::Tensor, torch::Tensor> splat_standalone(torch::Tensor& positions_raw, torch::Tensor& values); 
    // std::tuple<torch::Tensor, torch::Tensor> just_create_verts(torch::Tensor& positions_raw,  const bool return_indices_and_weights );  //creates splatting indices and splatting weights
    // std::shared_ptr<Lattice> expand(torch::Tensor& positions_raw, const int point_multiplier, const float noise_stddev, const bool expand_values );
    // std::tuple<std::shared_ptr<Lattice>, torch::Tensor, torch::Tensor, torch::Tensor> distribute(torch::Tensor& positions_raw, torch::Tensor& values, const bool reset_hashmap=true); 
    // torch::Tensor slice_standalone_with_precomputation(torch::Tensor& positions_raw, torch::Tensor& splatting_indices_tensor, torch::Tensor& splatting_weights_tensor);
    // std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> slice_standalone_no_precomputation(torch::Tensor& positions_raw); //slice at the position and don't use the m_matrix, but rather query the simplex and get the barycentric coordinates and all that. This is useful for when we slice at a different position than the one used for splatting
    // std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> gather_standalone_no_precomputation(torch::Tensor& positions_raw); //gathers the features of the neighbouring vertices and concats them all together, together with the barycentric weights. The output tensor will be size 1 x nr_positions x ( (m_pos_dim+1) x (val_full_dim +1) ). On each row we store sequencially the values of each vertex and then at the end we add the last m_pos_dim+1 barycentric weights
    // torch::Tensor gather_standalone_with_precomputation(torch::Tensor& positions_raw, torch::Tensor& splatting_indices_tensor, torch::Tensor& splatting_weights_tensor);
    // std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>  slice_classify_no_precomputation(torch::Tensor& positions_raw, torch::Tensor& delta_weights, torch::Tensor& linear_clasify_weight, torch::Tensor& linear_clasify_bias, const int nr_classes); //slices a lattices with some deltas applied to the barycentric coordinates and clasifies it in one go. Returns a tensor of class_logits of size 1 x nr_positions x nr_classes
    // torch::Tensor slice_classify_with_precomputation(torch::Tensor& positions_raw, torch::Tensor& delta_weights, torch::Tensor& linear_clasify_weight, torch::Tensor& linear_clasify_bias, const int nr_classes, torch::Tensor& splatting_indices_tensor, torch::Tensor& splatting_weights_tensor);
    
    // std::shared_ptr<Lattice> convolve_standalone(torch::Tensor& filter_bank); // convolves the lattice with a filter bank, creating a new values matrix. kernel_bank is a of size nr_filters x filter_extent x in_val_dim
    // std::shared_ptr<Lattice> convolve_im2row_standalone(torch::Tensor& filter_bank, const int dilation, std::shared_ptr<Lattice> lattice_neighbours, const bool flip_neighbours);
    // torch::Tensor im2row(std::shared_ptr<Lattice> lattice_neighbours, const int filter_extent, const int dilation, const bool flip_neighbours);

    // std::shared_ptr<Lattice> create_coarse_verts();  //creates another lattice which would be the result of splatting the positions/2. The values of the new coarse lattice are set to 0
    // std::shared_ptr<Lattice> create_coarse_verts_naive(torch::Tensor& positions_raw); //the previous one causes some positions to end up in empty space for some reason, so instead we use this to create vertices around all the positions, will be slower but possibly more correct

    // void slice_backwards_standalone_with_precomputation(torch::Tensor& positions_raw, const torch::Tensor& sliced_values_hom, const torch::Tensor& grad_sliced_values, torch::Tensor& splatting_indices_tensor, torch::Tensor& splatting_weights_tensor);
    // torch::Tensor row2im(const torch::Tensor& lattice_rowified,  const int dilation, const int filter_extent, const int nr_filters, std::shared_ptr<Lattice> lattice_neighbours );
    // void slice_backwards_elevated_verts_with_precomputation(const std::shared_ptr<Lattice> lattice_sliced_from, const torch::Tensor& grad_sliced_values, const int nr_verts_to_slice_from, torch::Tensor& splatting_indices_tensor, torch::Tensor& splatting_weights_tensor);
    // void slice_classify_backwards_with_precomputation(const torch::Tensor& grad_class_logits, torch::Tensor& positions_raw, torch::Tensor& initial_values,  torch::Tensor& delta_weights, torch::Tensor&  linear_clasify_weight, torch::Tensor& linear_clasify_bias, const int nr_classes, torch::Tensor& grad_lattice_values, torch::Tensor& grad_delta_weights,  torch::Tensor& grad_linear_clasify_weight, torch::Tensor& grad_linear_clasify_bias, torch::Tensor& splatting_indices_tensor, torch::Tensor& splatting_weights_tensor);
    // void gather_backwards_standalone_with_precomputation(const torch::Tensor& positions_raw, const torch::Tensor& grad_sliced_values, torch::Tensor& splatting_indices_tensor, torch::Tensor& splatting_weights_tensor);
    // void gather_backwards_elevated_standalone_with_precomputation(const std::shared_ptr<Lattice> lattice_gathered_from, const torch::Tensor& grad_sliced_values, torch::Tensor& splatting_indices_tensor, torch::Tensor& splatting_weights_tensor);

    //forward passes
    // std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> slice_with_collisions_standalone_no_precomputation(torch::Tensor& positions_raw, const bool should_precompute_tensors_for_backward); //slices but does not take into account collisions in the hash map
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> slice_with_collisions_standalone_no_precomputation_fast(torch::Tensor& lattice_values, torch::Tensor& positions_raw, const bool should_precompute_tensors_for_backward); //slices but does not take into account collisions in the hash map
    // std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> slice_with_collisions_standalone_no_precomputation_fast_mr_loop(const std::vector<torch::Tensor>& lattice_values_list, const std::vector<float> sigmas_list, torch::Tensor& positions_raw, const bool should_precompute_tensors_for_backward);
    // std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> slice_with_collisions_standalone_no_precomputation_fast_mr_monolithic(const torch::Tensor& lattice_values_monolithic, const std::vector<float> sigmas_list, const torch::Tensor& random_shift_monolithic, torch::Tensor& positions_raw, const bool require_lattice_values_grad, const bool require_positions_grad);
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> slice_with_collisions_standalone_no_precomputation_fast_mr_monolithic(const torch::Tensor& lattice_values_monolithic, const torch::Tensor& scale_factor, torch::Tensor& positions_raw, torch::Tensor& random_shift_monolithic, torch::Tensor& anneal_window, const bool concat_points, const float points_scaling, const bool require_lattice_values_grad, const bool require_positions_grad);


    //backwards passes 
    // void slice_backwards_standalone_with_precomputation_no_homogeneous(torch::Tensor& positions_raw, const torch::Tensor& grad_sliced_values, torch::Tensor& splatting_indices_tensor, torch::Tensor& splatting_weights_tensor);
    torch::Tensor slice_backwards_standalone_with_precomputation_no_homogeneous_mr_monolithic(torch::Tensor& positions_raw, const torch::Tensor& grad_sliced_values, torch::Tensor& splatting_indices_tensor, torch::Tensor& splatting_weights_tensor);
    std::tuple<torch::Tensor, torch::Tensor> slice_backwards_standalone_no_precomputation_no_homogeneous_mr_monolithic(torch::Tensor& positions_raw, torch::Tensor& lattice_values_monolithic, torch::Tensor& grad_sliced_values_monolithic, const torch::Tensor& scale_factor, torch::Tensor& random_shift_monolithic, torch::Tensor& anneal_window, const bool concat_points, const bool require_lattice_values_grad, const bool require_positions_grad);

    //double backward
    std::tuple<torch::Tensor, torch::Tensor> slice_double_back_from_positions_grad(const torch::Tensor& double_positions_grad, torch::Tensor& positions_raw, torch::Tensor& lattice_values_monolithic, torch::Tensor& grad_sliced_values_monolithic, const torch::Tensor& scale_factor, torch::Tensor& random_shift_monolithic, torch::Tensor& anneal_window, const bool concat_points);






    // std::shared_ptr<Lattice> clone_lattice();
    // Eigen::MatrixXd keys_to_verts();
    // Eigen::MatrixXd color_no_neighbours();
    // void increase_sigmas(const float stepsize);
    static torch::Tensor compute_scale_factor_tensor(const std::vector<float> sigmas_list, const int pos_dim);

    // Eigen::MatrixXf create_E_matrix(const int pos_dim); //create the elevation matrix that elevates from 3D to 4D. This gives the same results as the elevation that is done in cuda kernel only if the invStdDev is 1.0. So as if this line wouldn't exist: https://github.com/abadams/permutohedral/blob/e3538fbe2981be4bc0d67331fa811fd543c6f5da/cpu/permutohedral.h#L281
    // torch::Tensor elevate_points(torch::Tensor& positions_raw);
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> create_non_differentiable_indices_for_slice_with_collisions(torch::Tensor& positions_raw); //creates the rem0, rank and splatting indices for each position. This allows to create afterwards the elevated positions inside python (so it's tracked by autograd) using the E matrix and then get the barycentric coordinate also in a differentiable manner from the elevated and rem0 and rank. This allows to have the whole slicing code to be differentiable with gradients for both the lattice values and for the positions
    // void debug_repeted_weights(const Eigen::MatrixXf& points, const Eigen::MatrixXi& splatting_indices, const Eigen::MatrixXf& splatting_weights ); //debug why there are so many repeted splatting weights



    //getters
    // int val_dim();
    int pos_dim();
    int capacity();
    std::string name();
    // int nr_lattice_vertices(); //cannot obtain from here because we need the cuda part to perform a wait
    int get_filter_extent(const int neighborhood_size); //how many lattice points does a certain filter touch (eg for a pos_dim of 2 and neighbouhood of 1 we touch 7 verts, 6 for the hexagonal shape and 1 for the center)
    static int get_expected_filter_extent(const int neighborhood_size);
    torch::Tensor sigmas_tensor();
    torch::Tensor positions(); 
    // std::shared_ptr<HashTable> hash_table();
    // torch::Tensor values();
    static bool is_half_precision();
   

    //setters
    // void set_sigma(const float sigma);
    void set_name(const std::string name);
    // void set_values(const torch::Tensor& new_values, const bool sanity_check=true); //use this to set new values because it will also call update_impl to set the float pointers to the correct place in the implementation
    // void set_positions( const torch::Tensor& positions_raw ); // the positions that were used to create this lattice
    // void set_nr_lattice_vertices(const int nr_verts); //unwise to use it unless in very specific and weird cases in which you are doing something which allows collisions



   

private:
    // Lattice(const std::string config_file);
    // Lattice(const std::string config_file, const std::string name);
    // Lattice(Lattice* other);
    // void init_params(const std::string config_file);
     Lattice(const int capacity, const int pos_dim, const int nr_levels, const int nr_feat_per_level);
    // void set_and_check_input(torch::Tensor& positions_raw, torch::Tensor& values); //sets pos dim and val dim and then also checks that the positions and values are correct and we have sigmas for all posiitons dims
    void check_positions(const torch::Tensor& positions_raw);
    void check_positions_elevated(const torch::Tensor& positions_elevated);
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



  
};


