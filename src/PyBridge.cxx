#include "permutohedral_encoding/PyBridge.h"

#include <torch/extension.h>
#include "torch/torch.h"
#include "torch/csrc/utils/pybind.h"

//my stuff 
// #include "data_loaders/DataLoaderShapeNetPartSeg.h"
// #include "easy_pbr/Mesh.h"
// #include "easy_pbr/LabelMngr.h"
#include "permutohedral_encoding/Encoding.cuh"
// #include "instant_ngp_2/HashTable.cuh"
// #include "instant_ngp_2/InstantNGP.cuh"
// #include "instant_ngp_2/Sphere.cuh"
// #include "instant_ngp_2/VoxelGrid.cuh"
// #include "instant_ngp_2/OccupancyGrid.cuh"
// #include "instant_ngp_2/VolumeRendering.cuh"
// #include "instant_ngp_2/RaySampler.cuh"
// #include "instant_ngp_2/TrainParams.h"
// #include "instant_ngp_2/ModelParams.h"
// #include "instant_ngp_2/EvalParams.h"
// #include "instant_ngp_2/NGPGui.h"

// #include "easy_pbr/Viewer.h"


// https://pybind11.readthedocs.io/en/stable/advanced/cast/stl.html
// PYBIND11_MAKE_OPAQUE(std::vector<int>); //to be able to pass vectors by reference to functions and have things like push back actually work 
// PYBIND11_MAKE_OPAQUE(std::vector<float>, std::allocator<float> >);

namespace py = pybind11;




PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {


    //Lattice
    // py::module::import("torch");
    // py::object variable = (py::object) py::module::import("torch").attr("autograd").attr("Variable"); //from here but it segment faults https://pybind11.readthedocs.io/en/stable/advanced/misc.html
    // py::class_<HashTable, std::shared_ptr<HashTable>   > (m, "HashTable")
    // .def_readonly("m_values_tensor", &HashTable::m_values_tensor) //careful when using this because setting it and not using update_impl is a big bug
    // .def_readonly("m_keys_tensor", &HashTable::m_keys_tensor) //careful when using this because setting it and not using update_impl is a big bug
    // .def_readonly("m_nr_filled_tensor", &HashTable::m_nr_filled_tensor) ////careful when using this because setting it and not using update_impl is a big bug
    // .def("update_impl", &HashTable::update_impl)
    // .def("set_values", &HashTable::set_values)
    // ;

    py::class_<Lattice, std::shared_ptr<Lattice>   > (m, "Lattice")
    // py::class_<Lattice, std::shared_ptr<Lattice>   > (m, "Lattice", variable)
    // py::class_<Lattice, at::Tensor, std::shared_ptr<Lattice>   > (m, "Lattice")
    // py::class_<Lattice, torch::autograd::Variable, std::shared_ptr<Lattice>   > (m, "Lattice")
    // py::class_<Lattice, torch::autograd::Variable > (m, "Lattice")
    .def_static("create", &Lattice::create<const std::string> ) //for templated methods like this one we need to explicitly instantiate one of the arguments
    .def_static("create", &Lattice::create<const std::string, const std::string> ) //for templated methods like this one we need to explicitly instantiate one of the arguments
    // .def("init",  &Lattice::init )
    // .def("elevate_points",  &Lattice::elevate_points )
    // .def("create_E_matrix",  &Lattice::create_E_matrix )
    // .def("create_non_differentiable_indices_for_slice_with_collisions",  &Lattice::create_non_differentiable_indices_for_slice_with_collisions )
    // .def("debug_repeted_weights",  &Lattice::debug_repeted_weights )
    // .def("begin_splat",  &Lattice::begin_splat )
    //forward
    // .def("slice_with_collisions_standalone_no_precomputation", &Lattice::slice_with_collisions_standalone_no_precomputation )
    .def("slice_with_collisions_standalone_no_precomputation_fast", &Lattice::slice_with_collisions_standalone_no_precomputation_fast )
    // .def("slice_with_collisions_standalone_no_precomputation_fast_mr_loop", &Lattice::slice_with_collisions_standalone_no_precomputation_fast_mr_loop )
    .def("slice_with_collisions_standalone_no_precomputation_fast_mr_monolithic", &Lattice::slice_with_collisions_standalone_no_precomputation_fast_mr_monolithic )
    //backward
    // .def("slice_backwards_standalone_with_precomputation_no_homogeneous", &Lattice::slice_backwards_standalone_with_precomputation_no_homogeneous )
    .def("slice_backwards_standalone_with_precomputation_no_homogeneous_mr_monolithic", &Lattice::slice_backwards_standalone_with_precomputation_no_homogeneous_mr_monolithic )
    .def("slice_backwards_standalone_no_precomputation_no_homogeneous_mr_monolithic", &Lattice::slice_backwards_standalone_no_precomputation_no_homogeneous_mr_monolithic )
    .def("slice_double_back_from_positions_grad", &Lattice::slice_double_back_from_positions_grad )
    //other things
    // .def("get_filter_extent", &Lattice::get_filter_extent )
    // .def_static("get_expected_filter_extent", &Lattice::get_expected_filter_extent )
    // .def("val_dim", &Lattice::val_dim )
    .def("pos_dim", &Lattice::pos_dim )
    .def("name", &Lattice::name )
    // .def("nr_lattice_vertices", &Lattice::nr_lattice_vertices )
    .def("capacity", &Lattice::capacity )
    .def("positions", &Lattice::positions )
    .def_static("is_half_precision", &Lattice::is_half_precision )
    // .def_static("compute_scale_factor_tensor", &Lattice::compute_scale_factor_tensor )
    // .def("sigmas_tensor", &Lattice::sigmas_tensor)
    // .def("hash_table", &Lattice::hash_table)
    // .def("values", &Lattice::values)
    // .def("set_values", &Lattice::set_values, py::arg().noconvert(), py::arg("sanity_check") = true)
    // .def("set_positions", &Lattice::set_positions)
    // .def("set_nr_lattice_vertices", &Lattice::set_nr_lattice_vertices)
    // .def("clone_lattice", &Lattice::clone_lattice)
    // .def("clear", &Lattice::clear)
    // .def("clear_only_values", &Lattice::clear_only_values)
    // .def("increase_sigmas", &Lattice::increase_sigmas)
    // .def("set_sigma", &Lattice::set_sigma)
    // .def("set_sigmas_from_string", &Lattice::set_sigmas_from_string)
    ;

    // py::class_<InstantNGP, std::shared_ptr<InstantNGP>   > (m, "InstantNGP")
    // .def_static("create", &InstantNGP::create<const std::shared_ptr<easy_pbr::Viewer>& > ) //for templated methods like this one we need to explicitly instantiate one of the arguments
    // .def_static("random_rays_from_reel", &InstantNGP::random_rays_from_reel ) 
    // .def_static("rays_from_reprojection_reel", &InstantNGP::rays_from_reprojection_reel ) 
    // .def_static("spherical_harmonics", &InstantNGP::spherical_harmonics ) 
    // .def_static("update_errors_of_matching_indices", &InstantNGP::update_errors_of_matching_indices ) 
    // .def_static("meshgrid3d", &InstantNGP::meshgrid3d ) 
    // .def_static("low_discrepancy2d_sampling", &InstantNGP::low_discrepancy2d_sampling ) 
    // #ifdef HSDF_WITH_GL
    //   .def("render_atributes",  &InstantNGP::render_atributes )
    // #endif
    // ;

    // py::class_<Sphere> (m, "Sphere")
    // .def(py::init<const float, const Eigen::Vector3f>())
    // .def("ray_intersection", &Sphere::ray_intersection ) 
    // .def("rand_points_inside", &Sphere::rand_points_inside, py::arg("nr_points") ) 
    // .def_readwrite("m_center_tensor", &Sphere::m_center_tensor ) 
    // .def_readwrite("m_center", &Sphere::m_center ) 
    // .def_readwrite("m_radius", &Sphere::m_radius ) 
    // ;

    // py::class_<VoxelGrid> (m, "VoxelGrid")
    // // .def(py::init<const int, const int, const float, const Eigen::Vector3f>())
    // .def(py::init<>())
    // // .def("get_nr_voxels", &VoxelGrid::get_nr_voxels ) 
    // // .def("get_grid", &VoxelGrid::get_grid ) 
    // // .def("compute_center_points", &VoxelGrid::compute_center_points ) 
    // // .def("update_with_density", &VoxelGrid::update_with_density ) 
    // .def_static("slice", &VoxelGrid::slice ) 
    // .def_static("splat", &VoxelGrid::splat) 
    // .def_static("upsample_grid", &VoxelGrid::upsample_grid) 
    // .def_static("get_nr_of_mips", &VoxelGrid::get_nr_of_mips ) 
    // .def_static("get_size_for_mip", &VoxelGrid::get_size_for_mip ) 
    // .def_static("get_size_downsampled_grid", &VoxelGrid::get_size_downsampled_grid ) 
    // .def_static("get_size_upsampled_grid", &VoxelGrid::get_size_upsampled_grid ) 
    // .def_static("compute_grid_points", &VoxelGrid::compute_grid_points ) 
    // .def_static("slice_cpu", &VoxelGrid::slice_cpu ) 
    // ;


    // py::class_<OccupancyGrid> (m, "OccupancyGrid")
    // .def(py::init<const int, const float, const Eigen::Vector3f>())
    // // .def(py::init<>())
    // .def_static("make_grid_values", &OccupancyGrid::make_grid_values ) 
    // .def_static("make_grid_occupancy", &OccupancyGrid::make_grid_occupancy ) 
    // .def("set_grid_values", &OccupancyGrid::set_grid_values ) 
    // .def("set_grid_occupancy", &OccupancyGrid::set_grid_occupancy ) 
    // .def("get_grid_values", &OccupancyGrid::get_grid_values ) 
    // .def("get_grid_occupancy", &OccupancyGrid::get_grid_occupancy ) 
    // .def("get_nr_voxels", &OccupancyGrid::get_nr_voxels ) 
    // .def("get_nr_voxels_per_dim", &OccupancyGrid::get_nr_voxels_per_dim ) 
    // .def("compute_grid_points", &OccupancyGrid::compute_grid_points ) 
    // .def("compute_random_sample_of_grid_points", &OccupancyGrid::compute_random_sample_of_grid_points ) 
    // .def("create_cubes_for_occupied_voxels", &OccupancyGrid::create_cubes_for_occupied_voxels ) 
    // .def("check_occupancy", &OccupancyGrid::check_occupancy ) 
    // .def("update_with_density", &OccupancyGrid::update_with_density ) 
    // .def("update_with_density_random_sample", &OccupancyGrid::update_with_density_random_sample ) 
    // .def("update_with_sdf", &OccupancyGrid::update_with_sdf ) 
    // .def("update_with_sdf_random_sample", &OccupancyGrid::update_with_sdf_random_sample ) 
    // .def("compute_samples_in_occupied_regions", &OccupancyGrid::compute_samples_in_occupied_regions ) 
    // .def("compute_first_sample_start_of_occupied_regions", &OccupancyGrid::compute_first_sample_start_of_occupied_regions ) 
    // .def("advance_sample_to_next_occupied_voxel", &OccupancyGrid::advance_sample_to_next_occupied_voxel ) 
    // ;

    // py::class_<RaySamplesPacked> (m, "RaySamplesPacked")
    // .def(py::init<const int, const int>())
    // .def("get_valid_samples", &RaySamplesPacked::get_valid_samples ) 
    // .def_readwrite("samples_pos",  &RaySamplesPacked::samples_pos )
    // .def_readwrite("samples_pos_4d",  &RaySamplesPacked::samples_pos_4d )
    // .def_readwrite("samples_dirs",  &RaySamplesPacked::samples_dirs )
    // .def_readwrite("samples_z",  &RaySamplesPacked::samples_z )
    // .def_readwrite("samples_dt",  &RaySamplesPacked::samples_dt )
    // .def_readwrite("samples_sdf",  &RaySamplesPacked::samples_sdf )
    // .def("set_sdf", &RaySamplesPacked::set_sdf ) 
    // .def_readwrite("ray_start_end_idx",  &RaySamplesPacked::ray_start_end_idx )
    // .def_readwrite("ray_fixed_dt",  &RaySamplesPacked::ray_fixed_dt )
    // .def_readwrite("max_nr_samples",  &RaySamplesPacked::max_nr_samples )
    // .def_readwrite("cur_nr_samples",  &RaySamplesPacked::cur_nr_samples )
    // .def_readwrite("rays_have_equal_nr_of_samples",  &RaySamplesPacked::rays_have_equal_nr_of_samples )
    // .def_readwrite("fixed_nr_of_samples_per_ray",  &RaySamplesPacked::fixed_nr_of_samples_per_ray )
    // ;

    // py::class_<VolumeRendering> (m, "VolumeRendering")
    // .def(py::init<>())
    // .def_static("volume_render_nerf", &VolumeRendering::volume_render_nerf ) 
    // .def_static("compute_dt", &VolumeRendering::compute_dt ) 
    // .def_static("cumprod_alpha2transmittance", &VolumeRendering::cumprod_alpha2transmittance ) 
    // .def_static("integrate_rgb_and_weights", &VolumeRendering::integrate_rgb_and_weights ) 
    // .def_static("sdf2alpha", &VolumeRendering::sdf2alpha ) 
    // .def_static("sum_over_each_ray", &VolumeRendering::sum_over_each_ray ) 
    // .def_static("cumsum_over_each_ray", &VolumeRendering::cumsum_over_each_ray ) 
    // .def_static("compute_cdf", &VolumeRendering::compute_cdf )  
    // .def_static("importance_sample", &VolumeRendering::importance_sample )  
    // .def_static("combine_uniform_samples_with_imp", &VolumeRendering::combine_uniform_samples_with_imp )  
    // .def_static("compact_ray_samples", &VolumeRendering::compact_ray_samples )  
    // //backward passes
    // .def_static("volume_render_nerf_backward", &VolumeRendering::volume_render_nerf_backward ) 
    // .def_static("cumprod_alpha2transmittance_backward", &VolumeRendering::cumprod_alpha2transmittance_backward )  
    // .def_static("integrate_rgb_and_weights_backward", &VolumeRendering::integrate_rgb_and_weights_backward )  
    // .def_static("sum_over_each_ray_backward", &VolumeRendering::sum_over_each_ray_backward )  
    // ;

    // py::class_<RaySampler> (m, "RaySampler")
    // .def(py::init<>())
    // .def_static("compute_samples_bg", &RaySampler::compute_samples_bg ) 
    // ;



    //   //TrainParams
    // py::class_<TrainParams, std::shared_ptr<TrainParams>   > (m, "TrainParams", py::module_local())
    // .def_static("create", &TrainParams::create<const std::string> ) //for templated methods like this one we need to explicitly instantiate one of the arguments
    // .def("dataset_name",  &TrainParams::dataset_name )
    // .def("with_viewer",  &TrainParams::with_viewer )
    // .def("with_visdom",  &TrainParams::with_visdom )
    // .def("with_tensorboard",  &TrainParams::with_tensorboard )
    // .def("with_wandb",  &TrainParams::with_wandb )
    // .def("lr",  &TrainParams::lr )
    // .def("weight_decay",  &TrainParams::weight_decay )
    // .def("save_checkpoint",  &TrainParams::save_checkpoint )
    // .def("checkpoint_path",  &TrainParams::checkpoint_path )
    // ;

    // //EvalParams
    // py::class_<EvalParams, std::shared_ptr<EvalParams>   > (m, "EvalParams", py::module_local())
    // .def_static("create", &EvalParams::create<const std::string> ) //for templated methods like this one we need to explicitly instantiate one of the arguments
    // .def("dataset_name",  &EvalParams::dataset_name )
    // .def("with_viewer",  &EvalParams::with_viewer )
    // .def("checkpoint_path",  &EvalParams::checkpoint_path )
    // .def("do_write_predictions",  &EvalParams::do_write_predictions )
    // .def("output_predictions_path",  &EvalParams::output_predictions_path )
    // ;

    // //ModelParams
    // py::class_<ModelParams, std::shared_ptr<ModelParams>   > (m, "ModelParams", py::module_local())
    // .def_static("create", &ModelParams::create<const std::string> ) //for templated methods like this one we need to explicitly instantiate one of the arguments
    // .def("positions_mode",  &ModelParams::positions_mode )
    // .def("values_mode",  &ModelParams::values_mode )
    // .def("pointnet_channels_per_layer",  &ModelParams::pointnet_channels_per_layer )
    // .def("pointnet_start_nr_channels",  &ModelParams::pointnet_start_nr_channels )
    // .def("nr_downsamples",  &ModelParams::nr_downsamples )
    // .def("nr_blocks_down_stage",  &ModelParams::nr_blocks_down_stage )
    // .def("nr_blocks_bottleneck",  &ModelParams::nr_blocks_bottleneck )
    // .def("nr_blocks_up_stage",  &ModelParams::nr_blocks_up_stage )
    // .def("nr_levels_down_with_normal_resnet",  &ModelParams::nr_levels_down_with_normal_resnet )
    // .def("nr_levels_up_with_normal_resnet",  &ModelParams::nr_levels_up_with_normal_resnet )
    // .def("compression_factor",  &ModelParams::compression_factor )
    // .def("dropout_last_layer",  &ModelParams::dropout_last_layer )
    // // .def("experiment",  &ModelParams::experiment )
    // ;

    // //EvalParams
    // py::class_<NGPGui, std::shared_ptr<NGPGui>   > (m, "NGPGui", py::module_local())
    // .def_static("create",  &NGPGui::create<const std::shared_ptr<easy_pbr::Viewer>& > ) //for templated methods like this one we need to explicitly 
    // .def_readwrite("m_do_training",  &NGPGui::m_do_training )
    // .def_readwrite("m_control_view",  &NGPGui::m_control_view )
    // .def_readwrite("m_c2f_progress",  &NGPGui::m_c2f_progress )
    // .def_readwrite("m_nr_samples_per_ray",  &NGPGui::m_nr_samples_per_ray )
    // .def_readwrite("m_inv_s",  &NGPGui::m_inv_s )
    // .def_readwrite("m_inv_s_min",  &NGPGui::m_inv_s_min )
    // .def_readwrite("m_inv_s_max",  &NGPGui::m_inv_s_max )
    // .def_readwrite("m_volsdf_beta",  &NGPGui::m_volsdf_beta )
    // .def_readwrite("m_neus_cos_anneal",  &NGPGui::m_neus_cos_anneal )
    // .def_readwrite("m_neus_variance",  &NGPGui::m_neus_variance )
    // .def_readwrite("m_nerf_surface_beta",  &NGPGui::m_nerf_surface_beta )
    // .def_readwrite("m_nerf_surface_std",  &NGPGui::m_nerf_surface_std )
    // .def_readwrite("m_surface_prob_sigma",  &NGPGui::m_surface_prob_sigma )
    // .def_readwrite("m_surface_prob_height",  &NGPGui::m_surface_prob_height )
    // .def_readwrite("m_soft_opacity_sigma",  &NGPGui::m_soft_opacity_sigma )
    // .def_readwrite("m_sphere_y_shift",  &NGPGui::m_sphere_y_shift )
    // .def_readwrite("m_show_unisurf_weights",  &NGPGui::m_show_unisurf_weights )
    // .def_readwrite("m_show_volsdf_weights",  &NGPGui::m_show_volsdf_weights )
    // .def_readwrite("m_show_neus_weights",  &NGPGui::m_show_neus_weights )
    // .def_readwrite("m_show_nerf_surface_weights",  &NGPGui::m_show_nerf_surface_weights )
    // .def_readwrite("m_ray_origin_x_shift",  &NGPGui::m_ray_origin_x_shift )
    // .def_readwrite("m_ray_origin_y_shift",  &NGPGui::m_ray_origin_y_shift )
    // .def_readwrite("m_isolines_layer_z_coord",  &NGPGui::m_isolines_layer_z_coord )
    // .def_readwrite("m_compute_full_layer",  &NGPGui::m_compute_full_layer )
    // .def_readwrite("m_isoline_width",  &NGPGui::m_isoline_width )
    // .def_readwrite("m_distance_between_isolines",  &NGPGui::m_distance_between_isolines )
    // .def_readwrite("m_use_only_dense_grid",  &NGPGui::m_use_only_dense_grid )
    // .def_readwrite("m_spp",  &NGPGui::m_spp )
    // .def_readwrite("m_render_mitsuba",  &NGPGui::m_render_mitsuba )
    // .def_readwrite("m_mitsuba_res_x",  &NGPGui::m_mitsuba_res_x )
    // .def_readwrite("m_mitsuba_res_y",  &NGPGui::m_mitsuba_res_y )
    // .def_readwrite("m_use_controlable_frame",  &NGPGui::m_use_controlable_frame )
    // .def_readwrite("m_frame_idx_from_dataset",  &NGPGui::m_frame_idx_from_dataset )
    // .def_readwrite("m_render_full_img",  &NGPGui::m_render_full_img )
    // .def_readwrite("m_use_sphere_tracing",  &NGPGui::m_use_sphere_tracing )
    // .def_readwrite("m_nr_iters_sphere_tracing",  &NGPGui::m_nr_iters_sphere_tracing )
    // .def_readwrite("m_sphere_trace_agressiveness",  &NGPGui::m_sphere_trace_agressiveness )
    // .def_readwrite("m_sphere_trace_threshold_converged",  &NGPGui::m_sphere_trace_threshold_converged )
    // .def_readwrite("m_sphere_trace_push_in_gradient_dir",  &NGPGui::m_sphere_trace_push_in_gradient_dir )
    // .def_readwrite("m_chunk_size",  &NGPGui::m_chunk_size )
    // .def_readwrite("m_error_map_max",  &NGPGui::m_error_map_max )
    // ;


}



