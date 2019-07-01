// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include <cuda_runtime.h>

#include "array.hpp"

#include "../cuda4dnn/csl/kernels.hpp"
#include "../cuda4dnn/csl/kernel_utils.hpp"
#include "../cuda4dnn/csl/tensor.hpp"
#include "../cuda4dnn/csl/pointer.hpp"
#include "../cuda4dnn/csl/stream.hpp"

namespace cv { namespace dnn { namespace cuda4dnn { namespace csl  { namespace kernels {

    namespace raw {
        /* Reference: https://github.com/BVLC/caffe/blob/master/src/caffe/layers/concat_layer.cu */
        template <class T>
        __global__ void concat(
            span<T> output, std::size_t output_axis_size, std::size_t output_axis_offset,
            view<T> input, std::size_t input_axis_size, std::size_t concat_size)
        {
            /* we need to copy all the elements of input to some location in the output */
            for (auto idx : grid_stride_range(input.size())) {
                const auto total_concat_size = concat_size * input_axis_size;
                const auto concat_num = idx / total_concat_size;
                const auto concat_index = idx % total_concat_size;
                const auto top_index = concat_index +
                    (concat_num * output_axis_size + output_axis_offset) * concat_size;

                output[top_index] = input[idx];
            }
        }

        template <class T, std::size_t N>
        using array = utils::array<T, N>;

        template <class T, std::size_t N>
        __global__ void concat_with_offsets(
            span<T> output, array<int, N> out_strides, array<int, N> out_offset,
            view<T> input, array<int, N> in_strides)
        {
            for (auto i : grid_stride_range(input.size())) {
                /* compute input axis indices corresponding to element 'i' */
                array<std::size_t, N> in_index;
                in_index[0] = i / in_strides[0];
                for (int j = 1; j < N; j++)
                    in_index[j] = (i % in_strides[j - 1]) / in_strides[j];

                /* compute output axis indices corresponding to element 'i' */
                array<std::size_t, N> out_index;
                for (int j = 0; j < N; j++)
                    out_index[j] = out_offset[j] + in_index[j];

                /* compute output element number from output axis indices */
                std::size_t oidx = 0;
                for (int j = 0; j < N; j++)
                    oidx += out_index[j] * out_strides[j];

                output[oidx] = input[i];
            }
        }
    }

    template <class T>
    void concat(
        const Stream& stream,
        TensorSpan<T> output, std::size_t output_axis_offset,
        TensorView<T> input, std::size_t axis)
    {
        std::size_t concat_size = 1;
        for (int i = axis + 1; i < output.rank; i++)
            concat_size *= output.get_axis_size(i);

        std::size_t input_axis_size = input.get_axis_size(axis);
        std::size_t output_axis_size = output.get_axis_size(axis);

        auto policy = make_policy(raw::concat<T>, 0, stream);
        launch_kernel(raw::concat<T>, policy,
            output, output_axis_size, output_axis_offset,
            input, input_axis_size, concat_size);
    }

    template void concat<float>(const Stream&, TensorSpan<float>, std::size_t, TensorView<float>,  std::size_t);
    template void concat<double>(const Stream&, TensorSpan<double>, std::size_t, TensorView<double>, std::size_t);

    template <class T, std::size_t N> static
    void launch_concat_with_offsets_kernel(
        const Stream& stream,
        span<T> output, const std::vector<std::size_t>& outStride, const std::vector<std::size_t>& outOffset,
        view<T> input, const std::vector<std::size_t>& inStride)
    {
        CV_Assert(outStride.size() == N);
        CV_Assert(outOffset.size() == N);
        CV_Assert(inStride.size() == N);

        utils::array<int, N> outStride_k, outOffset_k, inStride_k;
        outStride_k.assign(std::begin(outStride), std::end(outStride));
        outOffset_k.assign(std::begin(outOffset), std::end(outOffset));
        inStride_k.assign(std::begin(inStride), std::end(inStride));

        auto kernel = raw::concat_with_offsets<T, N>;
        auto policy = make_policy(kernel, 0, stream);
        launch_kernel(kernel, policy, output, outStride_k, outOffset_k, input, inStride_k);
    }

    template <class T>
    void concat_with_offsets(
        const Stream& stream,
        TensorSpan<T> output, TensorView<T> input,
        const std::vector<std::size_t>& offsets)
    {
        CV_Assert(output.rank == input.rank);
        CV_Assert(output.rank >= 3 && output.rank <= 5);

        int rank = output.rank;
        auto inShape = input.shape();
        auto outShape = output.shape();

        std::vector<std::size_t> inStride(rank), outStride(rank);
        inStride.back() = 1;
        outStride.back() = 1;
        /* garbage, ..., garbage, 1 */

        std::copy(std::begin(inShape) + 1, std::end(inShape), std::begin(inStride));
        std::copy(std::begin(outShape) + 1, std::end(outShape), std::begin(outStride));
        /* dim[0], dim[1], ..., dim[-1], 1 */

        std::partial_sum(inStride.rbegin(), inStride.rend(), inStride.rbegin(), std::multiplies<int>());
        std::partial_sum(outStride.rbegin(), outStride.rend(), outStride.rbegin(), std::multiplies<int>());
        /* stride[0], stride[1], ..., stride[-2], 1 */

        if (offsets.size() != rank) {
            auto diff = rank - offsets.size();
            outStride.erase(outStride.begin(), outStride.begin() + diff);
            inStride.erase(inStride.begin(), inStride.begin() + diff);
        }

        if (rank == 5) {
            launch_concat_with_offsets_kernel<T, 5>(stream, output, outStride, offsets, input, inStride);
        } else if (rank == 4) {
            launch_concat_with_offsets_kernel<T, 4>(stream, output, outStride, offsets, input, inStride);
        } else if (rank == 3) {
            launch_concat_with_offsets_kernel<T, 3>(stream, output, outStride, offsets, input, inStride);
        }
    }

    template void concat_with_offsets(const Stream&, TensorSpan<float>, TensorView<float>, const std::vector<std::size_t>&);
    template void concat_with_offsets(const Stream&, TensorSpan<double>, TensorView<double>, const std::vector<std::size_t>&);

}}}}} /*  cv::dnn::cuda4dnn::csl::kernels */
