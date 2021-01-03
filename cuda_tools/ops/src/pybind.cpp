#include <torch/extension.h>
#include <vector>

using namespace at;

Tensor nms(Tensor boxes, float iou_threshold);

void roi_align_forward(Tensor input, Tensor rois, Tensor output, int aligned_height, int aligned_width, float spatial_scale, int sampling_ratio);

void roi_align_backward(Tensor grad_output, Tensor rois, Tensor grad_input, int aligned_height, int aligned_width, float spatial_scale, int sampling_ratio);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("nms", &nms, "nms (CPU/CUDA) ", py::arg("boxes"), py::arg("iou_threshold"));
    m.def("roi_align_forward", &roi_align_forward, "roi_align_forward",
            py::arg("input"),py::arg("rois"),py::arg("output"),py::arg("aligned_height"),
            py::arg("aligned_width"),py::arg("spatial_scale"),py::arg("sampling_ratio"));
    m.def("roi_align_backward",&roi_align_backward,"roi_align_backward",
            py::arg("grad_output"),py::arg("rois"),py::arg("grad_input"),py::arg("aligned_height"),
            py::arg("aligned_width"),py::arg("spatial_scale"),py::arg("sampling_ratio"));
}