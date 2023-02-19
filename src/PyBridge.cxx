#include "permutohedral_encoding/PyBridge.h"

#include <torch/extension.h>
#include "torch/torch.h"
#include "torch/csrc/utils/pybind.h"

//my stuff 
#include "permutohedral_encoding/Encoding.cuh"


namespace py = pybind11;




PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {


    py::class_<Encoding, std::shared_ptr<Encoding>   > (m, "Encoding")
    // .def_static("create", &Encoding::create<int, int, int, int> ) //for templated methods like this one we need to explicitly instantiate one of the arguments
    .def_static("create", &Encoding::create<> ) //for templated methods like this one we need to explicitly instantiate one of the arguments
    // .def_static("create", &Lattice::create<const std::string, const std::string> ) //for templated methods like this one we need to explicitly instantiate one of the arguments
    .def_static("test", &Encoding::test )
    //forward
    .def_static("slice_with_collisions_standalone_no_precomputation_fast_mr_monolithic", &Encoding::slice_with_collisions_standalone_no_precomputation_fast_mr_monolithic )
    //backward
    // .def("slice_backwards_standalone_with_precomputation_no_homogeneous_mr_monolithic", &Encoding::slice_backwards_standalone_with_precomputation_no_homogeneous_mr_monolithic )
    .def_static("slice_backwards_standalone_no_precomputation_no_homogeneous_mr_monolithic", &Encoding::slice_backwards_standalone_no_precomputation_no_homogeneous_mr_monolithic )
    .def_static("slice_double_back_from_positions_grad", &Encoding::slice_double_back_from_positions_grad )
    //other things
    .def_static("compute_scale_factor_tensor", &Encoding::compute_scale_factor_tensor )
    ;

    


}



