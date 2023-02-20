#include "permutohedral_encoding/PyBridge.h"

#include <torch/extension.h>
#include "torch/torch.h"
#include "torch/csrc/utils/pybind.h"

//my stuff 
#include "permutohedral_encoding/Encoding.cuh"


namespace py = pybind11;




PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

    m.def("create_encoding", &create_encoding);


 

    py::class_<EncodingInput> (m, "EncodingInput")
    .def(py::init<const torch::Tensor&, const torch::Tensor&, const torch::Tensor&, const bool, const bool>())
    .def_readwrite("m_lattice_values", &EncodingInput::m_lattice_values )
    .def_readwrite("m_positions_raw", &EncodingInput::m_positions_raw )
    .def_readwrite("m_anneal_window", &EncodingInput::m_anneal_window )
    .def_readwrite("m_require_lattice_values_grad", &EncodingInput::m_require_lattice_values_grad )
    .def_readwrite("m_require_positions_grad", &EncodingInput::m_require_positions_grad )
    ;
    py::class_<EncodingFixedParams> (m, "EncodingFixedParams")
    .def(py::init<const int, const int, const int, const int, const std::vector<float>&, const torch::Tensor&, const bool, const float>())
    ;
    // py::class_<EncodingBase,  std::shared_ptr<EncodingBase>   > (m, "EncodingBase")
    // ;


    py::class_<EncodingWrapper,  std::shared_ptr<EncodingWrapper>   > (m, "EncodingWrapper")
    .def_static("create", &EncodingWrapper::create<const int, const int, const EncodingFixedParams&> ) //for templated methods like this one we need to explicitly instantiate one of the arguments
    .def("forward", &EncodingWrapper::forward )
    .def("backward", &EncodingWrapper::backward )
    ;

    // py::class_<Encoding, EncodingBase, std::shared_ptr<Encoding>   > (m, "Encoding")
    // // .def_static("create", &Encoding::create<int, int, int, int> ) //for templated methods like this one we need to explicitly instantiate one of the arguments
    // .def_static("create", &Encoding::create<> ) //for templated methods like this one we need to explicitly instantiate one of the arguments
    // // .def_static("create", &Lattice::create<const std::string, const std::string> ) //for templated methods like this one we need to explicitly instantiate one of the arguments
    // .def_static("test", &Encoding::test )
    // //forward
    // .def_static("slice_with_collisions_standalone_no_precomputation_fast_mr_monolithic", &Encoding::slice_with_collisions_standalone_no_precomputation_fast_mr_monolithic )
    // //backward
    // // .def("slice_backwards_standalone_with_precomputation_no_homogeneous_mr_monolithic", &Encoding::slice_backwards_standalone_with_precomputation_no_homogeneous_mr_monolithic )
    // .def_static("slice_backwards_standalone_no_precomputation_no_homogeneous_mr_monolithic", &Encoding::slice_backwards_standalone_no_precomputation_no_homogeneous_mr_monolithic )
    // .def_static("slice_double_back_from_positions_grad", &Encoding::slice_double_back_from_positions_grad )
    // //other things
    // .def_static("compute_scale_factor_tensor", &Encoding::compute_scale_factor_tensor )
    // ;

    


}



