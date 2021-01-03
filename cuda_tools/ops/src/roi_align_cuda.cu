#include <cuda.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include "common_cuda_helper.hpp"

using at::Tensor;

template <typename T>
__global__ void roi_align_forward_cuda_kernel(
    const int nthreads, const T* input, const T* rois, T* output, const int pooled_height, const int pooled_width,
    const T spatial_scale, const int sampling_ratio, const int channels, const int height, const int width)
{
    CUDA_1D_KERNEL_LOOP(index, nthreads)
    {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    const T* offset_rois = rois + n * 5;
    int roi_batch_ind = offset_rois[0];

    // Do not using rounding; this implementation detail is critical
    T offset = 0.5;
    T roi_start_w = offset_rois[1] * spatial_scale - offset;
    T roi_start_h = offset_rois[2] * spatial_scale - offset;
    T roi_end_w = offset_rois[3] * spatial_scale - offset;
    T roi_end_h = offset_rois[4] * spatial_scale - offset;

    T roi_width = roi_end_w - roi_start_w;
    T roi_height = roi_end_h - roi_start_h;

    T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
    T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width);

    const T* offset_input = input + (roi_batch_ind * channels + c) * height * width;

    // We use roi_bin_grid to sample the grid and mimic integral
    int roi_bin_grid_h = (sampling_ratio > 0) ? sampling_ratio : static_cast<int>(ceil(bin_size_h));
    int roi_bin_grid_w = (sampling_ratio > 0) ? sampling_ratio : static_cast<int>(ceil(bin_size_w));

    // We do average pooling inside a bin
    const T count = max(roi_bin_grid_h * roi_bin_grid_w, 1);
    T output_val = 0.;
    for (int iy = 0; iy < roi_bin_grid_h; iy++)
    {
        const T y = roi_start_h + ph * bin_size_h + static_cast<T>(iy + .5f) * bin_size_h /static_cast<T>(roi_bin_grid_h);
        for (int ix = 0; ix < roi_bin_grid_w; ix++)
        {
          const T x = roi_start_w + pw * bin_size_w + static_cast<T>(ix + .5f) * bin_size_w / static_cast<T>(roi_bin_grid_w);
          T val = bilinear_interpolate(offset_input, height, width, y, x);
          output_val += val;
        }
    }
    output[index] = output_val / count;
  }
}

/*** Backward ***/
template <typename T>
__global__ void roi_align_backward_cuda_kernel(
    const int nthreads, const T* grad_output, const T* rois, T* grad_input, const int pooled_height,
    const int pooled_width, const T spatial_scale, const int sampling_ratio, const int channels, const int height, const int width)
{
    CUDA_1D_KERNEL_LOOP(index, nthreads)
    {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    const T grad_output_this_bin = grad_output[index];

    const T* offset_rois = rois + n * 5;
    int roi_batch_ind = offset_rois[0];
    T* offset_grad_input = grad_input + ((roi_batch_ind * channels + c) * height * width);

    // Do not using rounding; this implementation detail is critical
    T offset = 0.5;
    T roi_start_w = offset_rois[1] * spatial_scale - offset;
    T roi_start_h = offset_rois[2] * spatial_scale - offset;
    T roi_end_w = offset_rois[3] * spatial_scale - offset;
    T roi_end_h = offset_rois[4] * spatial_scale - offset;

    T roi_width = roi_end_w - roi_start_w;
    T roi_height = roi_end_h - roi_start_h;

    T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
    T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width);

    // We use roi_bin_grid to sample the grid and mimic integral
    int roi_bin_grid_h = (sampling_ratio > 0) ? sampling_ratio : static_cast<int>(ceil(bin_size_h));
    int roi_bin_grid_w = (sampling_ratio > 0) ? sampling_ratio : static_cast<int>(ceil(bin_size_w));

    // We do average (integral) pooling inside a bin
    const T count = roi_bin_grid_h * roi_bin_grid_w;  // e.g. = 4

    for (int iy = 0; iy < roi_bin_grid_h; iy++)
    {
        const T y = roi_start_h + ph * bin_size_h + static_cast<T>(iy + .5f) * bin_size_h / static_cast<T>(roi_bin_grid_h);
        for (int ix = 0; ix < roi_bin_grid_w; ix++)
        {
          const T x = roi_start_w + pw * bin_size_w + static_cast<T>(ix + .5f) * bin_size_w / static_cast<T>(roi_bin_grid_w);
          T w1, w2, w3, w4;
          int x_low, x_high, y_low, y_high;
          bilinear_interpolate_gradient(height, width, y, x, w1, w2, w3, w4,x_low, x_high, y_low, y_high);

          if (x_low >= 0 && x_high >= 0 && y_low >= 0 && y_high >= 0)
          {
            atomicAdd(offset_grad_input + y_low * width + x_low, grad_output_this_bin * w1 / count);
            atomicAdd(offset_grad_input + y_low * width + x_high, grad_output_this_bin * w2 / count);
            atomicAdd(offset_grad_input + y_high * width + x_low, grad_output_this_bin * w3 / count);
            atomicAdd(offset_grad_input + y_high * width + x_high, grad_output_this_bin * w4 / count);
          }
        }
    }
  }
}

void ROIAlignForwardCUDAKernelLauncher(Tensor input, Tensor rois, Tensor output,
                                       int aligned_height, int aligned_width,
                                       float spatial_scale, int sampling_ratio)
{
  int output_size = output.numel();
  int channels = input.size(1);
  int height = input.size(2);
  int width = input.size(3);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  roi_align_forward_cuda_kernel<<<GET_BLOCKS(output_size), THREADS_PER_BLOCK, 0, stream>>>(
                output_size, input.data_ptr<float>(),
                rois.data_ptr<float>(), output.data_ptr<float>(),
                aligned_height, aligned_width,
                static_cast<float>(spatial_scale), sampling_ratio, channels, height, width);
}

void ROIAlignBackwardCUDAKernelLauncher(Tensor grad_output, Tensor rois,
                                        Tensor grad_input, int aligned_height,
                                        int aligned_width, float spatial_scale,
                                        int sampling_ratio)
{
  int output_size = grad_output.numel();
  int channels = grad_input.size(1);
  int height = grad_input.size(2);
  int width = grad_input.size(3);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  roi_align_backward_cuda_kernel<<<GET_BLOCKS(output_size), THREADS_PER_BLOCK, 0, stream>>>(
                output_size, grad_output.data_ptr<float>(),
                rois.data_ptr<float>(), grad_input.data_ptr<float>(),
                aligned_height, aligned_width,
                static_cast<float>(spatial_scale), sampling_ratio, channels, height, width);
}