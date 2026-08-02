#include <torch/extension.h>

#include <cstdint>
#include <string>
#include <vector>

namespace
{

at::ScalarType parse_dtype(const std::string &dtype)
{
  if (dtype == "uint8") return at::kByte;
  if (dtype == "float16") return at::kHalf;
  if (dtype == "bfloat16") return at::kBFloat16;
  if (dtype == "float32") return at::kFloat;
  TORCH_CHECK(false, "unsupported external CUDA tensor dtype: ", dtype);
}

torch::Tensor tensor_from_cuda_ptr(std::uint64_t device_ptr,
                                   const std::vector<std::int64_t> &sizes,
                                   const std::vector<std::int64_t> &strides,
                                   const std::string &dtype,
                                   int device_index)
{
  TORCH_CHECK(device_ptr != 0, "external CUDA pointer must be non-zero");
  TORCH_CHECK(!sizes.empty(), "external CUDA tensor sizes cannot be empty");
  TORCH_CHECK(sizes.size() == strides.size(),
              "sizes and strides must have equal rank");
  for (std::size_t index = 0; index < sizes.size(); ++index)
  {
    TORCH_CHECK(sizes[index] > 0, "tensor sizes must be positive");
    TORCH_CHECK(strides[index] >= 0, "negative strides are not supported");
  }

  const auto options = torch::TensorOptions()
                           .dtype(parse_dtype(dtype))
                           .device(torch::Device(torch::kCUDA, device_index));

  // Tensor 只是 daemon-owned allocation 的 client view。no-op deleter 保证
  // Tensor 析构不会越权 cudaFree；owner 最终由 BaM MDS service 释放。
  auto no_op_deleter = [](void *) {};
  return torch::from_blob(reinterpret_cast<void *>(device_ptr),
                          sizes,
                          strides,
                          no_op_deleter,
                          options);
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module)
{
  module.doc() =
      "Wrap a daemon-owned CUDA IPC pointer as a non-owning PyTorch Tensor";
  module.def("tensor_from_cuda_ptr",
             &tensor_from_cuda_ptr,
             pybind11::arg("device_ptr"),
             pybind11::arg("sizes"),
             pybind11::arg("strides"),
             pybind11::arg("dtype"),
             pybind11::arg("device_index") = 0);
}
