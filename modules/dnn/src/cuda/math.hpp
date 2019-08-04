// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_SRC_CUDA_MATH_HPP
#define OPENCV_DNN_SRC_CUDA_MATH_HPP

#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace cv { namespace dnn { namespace cuda4dnn { namespace csl { namespace device {

    template <class T> __device__ T abs(T val) { return (val < T(0) ? -val : val); }
    template <> inline __device__ float abs(float val) { return fabsf(val); }
    template <> inline __device__ double abs(double val) { return fabs(val); }

    template <class T> __device__ T exp(T val);
    template <> inline __device__ __half exp(__half val) { return hexp(val); }
    template <> inline __device__ __half2 exp(__half2 val) { return h2exp(val); }
    template <> inline __device__ float exp(float val) { return expf(val); }
    template <> inline __device__ double exp(double val) { return ::exp(val); }

    template <class T> __device__ T expm1(T val);
    template <> inline __device__ __half expm1(__half val) { return hexp(val) + __half(1); }
    template <> inline __device__ __half2 expm1(__half2 val) { return h2exp(val) + __half2(1, 1); }
    template <> inline __device__ float expm1(float val) { return expm1f(val); }
    template <> inline __device__ double expm1(double val) { return ::expm1(val); }

    template <class T> __device__ T max(T x, T y) { return (x > y ? x : y); }
    template <> inline __device__ float max(float x, float y) { return fmaxf(x, y); }
    template <> inline __device__ double max(double x, double y) { return fmax(x, y); }

    template <class T> __device__ T min(T x, T y) { return (x > y ? y : x); }
    template <> inline __device__ float min(float x, float y) { return fminf(x, y); }
    template <> inline __device__ double min(double x, double y) { return fmin(x, y); }

    template <class T> __device__ T log1p(T val);
    template <> inline __device__ __half log1p(__half val) { return hlog(val) + __half(1); }
    template <> inline __device__ __half2 log1p(__half2 val) { return h2log(val) + __half2(1, 1); }
    template <> inline __device__ float log1p(float val) { return log1pf(val); }
    template <> inline __device__ double log1p(double val) { return ::log1p(val); }

    template <class T> __device__ T log1pexp(T val);
    template <> inline __device__ double log1pexp(double val) {
        if (val <= -37)
            return exp(val);
        else if (-37 < val && val <= 18)
            return log1p(exp(val));
        else if (18 < val && val <= 33.3)
            return val + exp(-val);
        else
            return val;
    }
    template <> inline __device__ float log1pexp(float val) { return log1pexp<double>(val); }
    template <> inline __device__ __half log1pexp(__half val) { return log1pexp<double>(val); }

    template <class T> __device__ T tanh(T val);
    template <> inline __device__ __half tanh(__half val) { return tanhf(val); }
    template <> inline __device__ float tanh(float val) { return tanhf(val); }
    template <> inline __device__ double tanh(double val) { return ::tanh(val); }

    template <class T> __device__ T pow(T val, T exp);
    template <> inline __device__ __half pow(__half val, __half exp) { return powf(val, exp); }
    template <> inline __device__ float pow(float val, float exp) { return powf(val, exp); }
    template <> inline __device__ double pow(double val, double exp) { return ::pow(val, exp); }

    template <class T> __device__ T sqrt(T val);
    template <> inline __device__ __half sqrt(__half val) { return hsqrt(val); }
    template <> inline __device__ __half2 sqrt(__half2 val) { return h2sqrt(val); }
    template <> inline __device__ float sqrt(float val) { return sqrtf(val); }
    template <> inline __device__ double sqrt(double val) { return ::sqrt(val); }

    template <class T> __device__ T rsqrt(T val);
    template <> inline __device__ __half rsqrt(__half val) { return hrsqrt(val); }
    template <> inline __device__ __half2 rsqrt(__half2 val) { return h2rsqrt(val); }
    template <> inline __device__ float rsqrt(float val) { return rsqrtf(val); }
    template <> inline __device__ double rsqrt(double val) { return ::rsqrt(val); }

    template <class T> __device__ T sigmoid(T val) { return T(1) / (T(1) + exp(-val)); }

    template <class T> __device__ T clamp(T value, T lower, T upper) { return min(max(value, lower), upper); }

}}}}} /*  cv::dnn::cuda4dnn::csl::device */

#endif /* OPENCV_DNN_SRC_CUDA_MATH_HPP */
