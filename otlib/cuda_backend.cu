#include "cuda_backend.h"

#ifdef OTMAP_HAS_CUDA_BACKEND

#include <cuda_runtime.h>
#include <iostream>

namespace otmap {

namespace {

__global__ void scale_kernel(double* data, std::size_t length, double scale)
{
  std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < length)
  {
    data[idx] *= scale;
  }
}

inline bool check_cuda(cudaError_t err, const char* message)
{
  if(err != cudaSuccess)
  {
    std::cerr << "CUDA error: " << message << ": " << cudaGetErrorString(err) << std::endl;
    return false;
  }
  return true;
}

} // namespace

bool cuda_backend_available()
{
  int device_count = 0;
  auto result = cudaGetDeviceCount(&device_count);
  return (result == cudaSuccess) && device_count > 0;
}

bool cuda_scale_array(double* host_data, std::size_t length, double scale)
{
  if(length == 0 || host_data == nullptr)
    return false;

  double* device_ptr = nullptr;
  std::size_t bytes = length * sizeof(double);

  if(!check_cuda(cudaMalloc(&device_ptr, bytes), "cudaMalloc"))
    return false;

  if(!check_cuda(cudaMemcpy(device_ptr, host_data, bytes, cudaMemcpyHostToDevice), "cudaMemcpy H2D"))
  {
    cudaFree(device_ptr);
    return false;
  }

  constexpr int block_size = 256;
  int grid_size = static_cast<int>((length + block_size - 1) / block_size);
  scale_kernel<<<grid_size, block_size>>>(device_ptr, length, scale);
  auto kernel_err = cudaGetLastError();
  if(kernel_err != cudaSuccess)
  {
    std::cerr << "CUDA kernel launch failed: " << cudaGetErrorString(kernel_err) << std::endl;
    cudaFree(device_ptr);
    return false;
  }

  if(!check_cuda(cudaMemcpy(host_data, device_ptr, bytes, cudaMemcpyDeviceToHost), "cudaMemcpy D2H"))
  {
    cudaFree(device_ptr);
    return false;
  }

  cudaFree(device_ptr);
  return true;
}

} // namespace otmap

#endif // OTMAP_HAS_CUDA_BACKEND
