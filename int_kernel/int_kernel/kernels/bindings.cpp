// PyTorch headers
#include <torch/extension.h>
#include <ATen/ATen.h>

// Local headers
#include "include/int_kernel.h"
#include "s8t_s8n_f16t_kernel.h"
#include "s4t_s4n_f16t_kernel.h"
#include "s8t_s4n_f16t_kernel.h"
using namespace torch::indexing;
namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // m.def("w4a4_gemm", &w4a4_gemm_with_quant, 
    //     "W4A4 GEMM kernel with quantization fusion (CUDA)",
    //     py::arg("input"),
    //     py::arg("weight"),
    //     py::arg("input_scale"),
    //     py::arg("input_zero_point"),
    //     py::arg("weight_scale"),
    //     py::arg("weight_zero_point"),
    //     py::arg("output_scale"),
    //     py::arg("output_zero_point"));
        
    // m.def("w4a8_gemm", &w4a8_gemm_with_quant,
    //     "W4A8 GEMM kernel with quantization fusion (CUDA)",
    //     py::arg("input"),
    //     py::arg("weight"),
    //     py::arg("input_scale"),
    //     py::arg("input_zero_point"),
    //     py::arg("weight_scale"),
    //     py::arg("weight_zero_point"),
    //     py::arg("output_scale"),
    //     py::arg("output_zero_point"));
        
    // m.def("w8a8_gemm", &w8a8_gemm_with_quant,
    //     "W8A8 GEMM kernel with quantization fusion (CUDA)",
    //     py::arg("input"),
    //     py::arg("weight"),
    //     py::arg("input_scale"),
    //     py::arg("input_zero_point"),
    //     py::arg("weight_scale"),
    //     py::arg("weight_zero_point"),
    //     py::arg("output_scale"),
    //     py::arg("output_zero_point"));

    m.def("s8t_s8n_f16t_gemm", &s8t_s8n_f16t_gemm,
        "S8T S8N F16T GEMM kernel with quantization fusion (CUDA)",
        py::arg("input"),
        py::arg("weight"));
    m.def("s4t_s4n_f16t_gemm", &s4t_s4n_f16t_gemm,
        "S4T S4N F16T GEMM kernel with quantization fusion (CUDA)",
        py::arg("input"),
        py::arg("weight"));
    m.def("s8t_s4n_f16t_gemm", &s8t_s4n_f16t_gemm,
        "S8T S4N F16T GEMM kernel with quantization fusion (CUDA)",
        py::arg("input"),
        py::arg("weight"));
} 