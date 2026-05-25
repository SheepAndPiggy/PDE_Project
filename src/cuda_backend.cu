#include "cuda_backend.hpp"

#include <cuda_runtime.h>

namespace cuda_backend {

int device_count()
{
    int count = 0;
    const cudaError_t status = cudaGetDeviceCount(&count);
    return status == cudaSuccess ? count : 0;
}

int runtime_version()
{
    int version = 0;
    const cudaError_t status = cudaRuntimeGetVersion(&version);
    return status == cudaSuccess ? version : 0;
}

} // namespace cuda_backend
