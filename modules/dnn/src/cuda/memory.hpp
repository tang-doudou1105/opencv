// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_SRC_CUDA_MEMORY_HPP
#define OPENCV_DNN_SRC_CUDA_MEMORY_HPP

#include <cuda_runtime.h>

namespace cv { namespace dnn { namespace cuda4dnn { namespace csl { namespace device {

template <class T>
__device__ T load_ldg(const T& src) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 350)
    return __ldg(&src);
#else
    return src;
#endif
}

template <class T>
__device__ T load_ldg(const T* src) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 350)
    return __ldg(src);
#else
    return *src;
#endif
}

#define DEVICE_STATIC_INTRINSIC_QUALIFIERS  static __device__ __forceinline__

#if (defined(_MSC_VER) && defined(_WIN64)) || defined(__LP64__)
#define PXL_GLOBAL_PTR   "l"
#else
#define PXL_GLOBAL_PTR   "r"
#endif

DEVICE_STATIC_INTRINSIC_QUALIFIERS void __prefetch_global_l1(const void* const ptr)
{
    asm("prefetch.global.L1 [%0];" : : PXL_GLOBAL_PTR(ptr));
}

DEVICE_STATIC_INTRINSIC_QUALIFIERS void __prefetch_global_uniform(const void* const ptr)
{
    asm("prefetchu.L1 [%0];" : : PXL_GLOBAL_PTR(ptr));
}

DEVICE_STATIC_INTRINSIC_QUALIFIERS void __prefetch_global_l2(const void* const ptr)
{
    asm("prefetch.global.L2 [%0];" : : PXL_GLOBAL_PTR(ptr));
}

}}}}} /* namespace cv::dnn::cuda4dnn::csl::device */

#endif /* OPENCV_DNN_SRC_CUDA_MEMORY_HPP */
