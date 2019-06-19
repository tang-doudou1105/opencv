/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
// Copyright (C) 2017, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "../precomp.hpp"
#include "layers_common.hpp"
#include "../op_cuda.hpp"

#ifdef HAVE_CUDA
#include "../cuda4dnn/csl/tensor.hpp"
using namespace cv::dnn::cuda4dnn;
#endif

namespace cv
{
namespace dnn
{

class SplitLayerImpl CV_FINAL : public SplitLayer
{
public:
    SplitLayerImpl(const LayerParams &params)
    {
        setParamsFrom(params);
        //TODO: maybe "top_count" param is useless because it can be determined by output connections number
        if (params.has("top_count"))
        {
            outputsCount = params.get<int>("top_count");
            CV_Assert(outputsCount >= 0);
        }
        else
        {
            outputsCount = -1;
        }
    }

    bool getMemoryShapes(const std::vector<MatShape> &inputs,
                         const int requiredOutputs,
                         std::vector<MatShape> &outputs,
                         std::vector<MatShape> &internals) const CV_OVERRIDE
    {
        CV_Assert(inputs.size() == 1);

        Layer::getMemoryShapes(inputs, max(1, outputsCount >= 0 ? outputsCount : requiredOutputs),
                               outputs, internals);
        return false;
    }

    bool supportBackend(int backendId) CV_OVERRIDE
    {
        return backendId == DNN_BACKEND_OPENCV ||
               (backendId == DNN_BACKEND_CUDA && haveCUDA());
    }

    void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays internals_arr) CV_OVERRIDE
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        std::vector<Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);
        for (size_t i = 0; i < outputs.size(); i++)
        {
            CV_Assert(inputs[0].total() == outputs[i].total());
            inputs[0].copyTo(outputs[i]);
        }
    }

#ifdef HAVE_CUDA
    void forwardCUDA(
        std::vector<cv::Ptr<BackendWrapper>>& inputs,
        std::vector<cv::Ptr<BackendWrapper>>& outputs,
        csl::Workspace& workspace
    )
    {
        CV_UNUSED(workspace);

        auto input_wrapper = inputs[0].dynamicCast<CUDABackendWrapperFP32>();
        auto input = input_wrapper->getView();

        for (std::size_t i = 0; i < outputs.size(); i++)
        {
            auto output_wrapper = outputs[i].dynamicCast<CUDABackendWrapperFP32>();
            auto output = output_wrapper->getSpan();

            if (input.get() != output.get())
                csl::tensor_ops::copy(stream, output, input);
        }
    }

    void initCUDA(
        csl::Stream stream_,
        csl::cublas::Handle cublas_handle,
        csl::cudnn::Handle cudnn_handle,
        std::size_t& scratch_mem_in_bytes
    )
    {
        stream = std::move(stream_);
    }

    csl::Stream stream;
#endif

};

Ptr<SplitLayer> SplitLayer::create(const LayerParams& params)
{
    return Ptr<SplitLayer>(new SplitLayerImpl(params));
}

}
}
