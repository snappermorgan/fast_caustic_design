#include "cuda_backend.h"

#ifndef OTMAP_HAS_CUDA_BACKEND

namespace otmap {

bool cuda_backend_available()
{
  return false;
}

bool cuda_scale_array(double*, std::size_t, double)
{
  return false;
}

} // namespace otmap

#endif // OTMAP_HAS_CUDA_BACKEND
