#include <torch/extension.h>
#include <vector>

using namespace at;

void ROIAlignForwardCUDAKernelLauncher(Tensor input, Tensor rois, Tensor output, int aligned_height, int aligned_width,
                                        float spatial_scale, int sampling_ratio);

void ROIAlignBackwardCUDAKernelLauncher(Tensor grad_output, Tensor rois, Tensor grad_input, int aligned_height,
                                        int aligned_width, float spatial_scale, int sampling_ratio);

void roi_align_forward(Tensor input, Tensor rois, Tensor output, int aligned_height, int aligned_width,
                        float spatial_scale, int sampling_ratio)
{
    ROIAlignForwardCUDAKernelLauncher(input, rois, output, aligned_height, aligned_width, spatial_scale, sampling_ratio);
}

void roi_align_backward(Tensor grad_output, Tensor rois, Tensor grad_input, int aligned_height, int aligned_width,
                         float spatial_scale, int sampling_ratio)
{
    ROIAlignBackwardCUDAKernelLauncher(grad_output, rois, grad_input, aligned_height, aligned_width, spatial_scale, sampling_ratio);
}
