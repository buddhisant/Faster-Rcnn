import torch
import numpy as np
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair
from cuda_tools import ops

class RoIAlignFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx,input,rois,output_size,spatial_scale,sampling_ratio=0):
        ctx.output_size=_pair(output_size)
        ctx.spatial_scale=spatial_scale
        ctx.sampling_ratio=sampling_ratio
        ctx.input_shape=input.size()

        output_shape = (rois.size(0),input.size(1),ctx.output_size[0],ctx.output_size[1])
        output = input.new_zeros(output_shape)

        ops.roi_align_forward(input,rois,output,ctx.output_size[0],ctx.output_size[1],ctx.spatial_scale,ctx.sampling_ratio)
        ctx.save_for_backward(rois)

        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        rois = ctx.saved_tensors[0]
        grad_input = grad_output.new_zeros(ctx.input_shape)

        ops.roi_align_backward(
            grad_output,
            rois,
            grad_input,
            aligned_height=ctx.output_size[0],
            aligned_width=ctx.output_size[1],
            spatial_scale=ctx.spatial_scale,
            sampling_ratio=ctx.sampling_ratio)
        return grad_input, None, None, None, None

roi_align=RoIAlignFunction.apply

class RoIAlign(torch.nn.Module):
    def __init__(self,output_size, spatial_scale=1.0, sampling_ratio=0):
        super(RoIAlign, self).__init__()
        self.output_size=_pair(output_size)
        self.spatial_scale=float(spatial_scale)
        self.sampling_ratio=int(sampling_ratio)

    def forward(self, input, rois):
        return roi_align(input,rois,self.output_size,self.spatial_scale,self.sampling_ratio)

if __name__=="__main__":
    input=torch.tensor([[[1,2],[3,4]],[[4,3],[2,1]]],dtype=torch.float)
    input=input.cuda()
    input=input.view(1,2,2,2)
    input.requires_grad=True

    M=RoIAlign(2,spatial_scale=1,sampling_ratio=2)
    rois=torch.tensor([[0,0,0,1,1]],dtype=torch.float)
    rois=rois.cuda()
    rois=rois.view(1,5)

    output=M(input,rois)
    output.backward(torch.ones_like(output))
    print(output)
    print(input.grad)