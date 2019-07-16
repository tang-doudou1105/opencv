// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_CUDA4DNN_PRIMITIVES_SLICE_HPP
#define OPENCV_DNN_CUDA4DNN_PRIMITIVES_SLICE_HPP

#include "../../op_cuda.hpp"

#include "../csl/stream.hpp"
#include "../csl/kernels.hpp"

#include <opencv2/core.hpp>

#include <cstddef>
#include <vector>
#include <utility>

namespace cv { namespace dnn { namespace cuda4dnn {

    template <class T>
    class SliceOp final : public CUDABackendNode {
    public:
        using wrapper_type = GetCUDABackendWrapperType<T>;

        /* offsets is indexed by output number and each subvector is indexed by axis number */
        SliceOp(csl::Stream stream_, std::vector<std::vector<std::size_t>> offsets)
            : stream(std::move(stream_)), offsets(std::move(offsets))
        {
        }

        void forward(
            std::vector<cv::Ptr<BackendWrapper>>& inputs,
            std::vector<cv::Ptr<BackendWrapper>>& outputs,
            csl::Workspace& workspace) override
        {
            CV_Assert(inputs.size() == 1);

            auto input_wrapper = inputs[0].dynamicCast<wrapper_type>();
            auto input = input_wrapper->getView();

            for (int i = 0; i < outputs.size(); ++i)
            {
                auto output_wrapper = outputs[i].dynamicCast<wrapper_type>();
                auto output = output_wrapper->getSpan();

                csl::kernels::slice<T>(stream, output, input, offsets[i]);
            }
        }

    private:
        csl::Stream stream;
        std::vector<std::vector<std::size_t>> offsets;
    };

}}} /* namespace cv::dnn::cuda4dnn */

#endif /* OPENCV_DNN_CUDA4DNN_PRIMITIVES_SLICE_HPP */
