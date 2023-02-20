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


    py::class_<EncodingWrapper,  std::shared_ptr<EncodingWrapper>   > (m, "EncodingWrapper")
    .def_static("create", &EncodingWrapper::create<const int, const int, const EncodingFixedParams&> ) //for templated methods like this one we need to explicitly instantiate one of the arguments
    .def("forward", &EncodingWrapper::forward )
    .def("backward", &EncodingWrapper::backward )
    .def("double_backward_from_positions", &EncodingWrapper::double_backward_from_positions )
    ;

    

    


}



