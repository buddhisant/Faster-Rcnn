import torch
import config as cfg
import utils
import math

import torch.nn.functional as F
from torch import nn
from loss import Loss
from inference import Inference

class Rpn(torch.nn.Module):
    def __init__(self,is_train=True):
        super(Rpn, self).__init__()
        self.is_train=is_train
        self.base_anchors=utils.compute_base_anchors()
        self.num_anchors=len(self.base_anchors[0])
        self.feat_channels=cfg.fpn_channels
        self.rpn_conv = nn.Conv2d(self.feat_channels,self.feat_channels,3,padding=1)
        self.rpn_cls = nn.Conv2d(self.feat_channels,self.num_anchors,1)
        self.rpn_reg = nn.Conv2d(self.feat_channels,self.num_anchors*4,1)

        loss_config={}
        loss_config["sample_nums"]=cfg.rpn_nums
        loss_config["pos_fraction"]=cfg.rpn_pos_fraction
        loss_config["encode_mean"]=cfg.rpn_encode_mean
        loss_config["encode_std"]=cfg.rpn_encode_std
        loss_config["num_classes"]=1
        loss_config["neg_th"]=cfg.rpn_neg_th
        loss_config["pos_th"]=cfg.rpn_pos_th
        loss_config["th_by_gt"]=cfg.rpn_gt_pos_th

        self.loss=Loss(loss_config,is_rpn=True)

        inference_config={}
        inference_config["encode_mean"]=cfg.rpn_encode_mean
        inference_config["encode_std"]=cfg.rpn_encode_std
        inference_config["num_classes"]=1
        inference_config["nms_pre"]=cfg.rpn_nms_pre if self.is_train else cfg.rpn_nms_pre_test
        inference_config["nms_threshold"]=cfg.rpn_nms_threshold
        inference_config["nms_post"]=cfg.rpn_nms_post
        inference_config["cls_output_channels"]=1
        self.inference=Inference(inference_config,is_rpn=True)

        for m in self.modules():
            if(isinstance(m,nn.Conv2d)):
                nn.init.normal_(m.weight,0,0.01)
                nn.init.constant_(m.bias,0)

    def compute_valid_flag(self, pad_img_shape, scales, num_anchors, device):
        valid_flag_per_img=[]
        for i, scale in enumerate(scales):
            stride=float(cfg.fpn_strides[i])
            h_fpn = scale[0]
            w_fpn = scale[1]
            h_valid = math.ceil(pad_img_shape[0]/stride)
            w_valid = math.ceil(pad_img_shape[1]/stride)

            y_valid = torch.zeros((h_fpn,), device=device, dtype=torch.bool)
            x_valid = torch.zeros((w_fpn,), device=device, dtype=torch.bool)
            x_valid[:w_valid] = 1
            y_valid[:h_valid] = 1

            y_valid,x_valid = torch.meshgrid(y_valid,x_valid)
            y_valid=y_valid.reshape(-1)
            x_valid=x_valid.reshape(-1)
            valid_flag_per_level=y_valid&x_valid
            valid_flag_per_level=valid_flag_per_level.view(-1,1).repeat(1,num_anchors).view(-1)
            valid_flag_per_img.append(valid_flag_per_level)
        return valid_flag_per_img

    def forward(self,features,res_img_shape,pad_img_shape,gt_bboxes=None):
        cls_preds=[]
        reg_preds=[]
        for feat in features:
            feat=self.rpn_conv(feat)
            feat=F.relu(feat,inplace=True)
            cls_preds.append(self.rpn_cls(feat))
            reg_preds.append(self.rpn_reg(feat))

        dtype = cls_preds[0].dtype
        device = cls_preds[0].device
        scales = [cls_pred.shape[-2:] for cls_pred in cls_preds]
        anchors = utils.compute_anchors(self.base_anchors, scales, device, dtype)

        proposal=self.inference(cls_preds,reg_preds,anchors,res_img_shape)

        rpn_loss=None
        if(self.is_train):
            valids = []
            num_anchors = self.base_anchors[0].size(0)
            for shape in pad_img_shape:
                valid_per_img = self.compute_valid_flag(shape, scales, num_anchors, device)
                valid_per_img = torch.cat(valid_per_img, dim=0)
                valids.append(valid_per_img)

            anchors=torch.cat(anchors,dim=0)
            targets=self.loss.compute_targets(anchors,valids,gt_bboxes)
            rpn_loss=self.loss(cls_preds, reg_preds, targets)

        return rpn_loss, proposal

