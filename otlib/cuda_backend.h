#pragma once

#include <cstddef>

namespace otmap {

// Returns true if a CUDA device is available at runtime.
bool cuda_backend_available();

// Scales an array in-place on the GPU. Returns true on success.
bool cuda_scale_array(double* host_data, std::size_t length, double scale);

} // namespace otmap
