import torch
import math
import config as cfg

from torch import nn
from resnet import resNet
from fpn import PyramidFeatures
from rpn import Rpn
from loss import Loss
from inference import Inference
from cuda_tools import RoIAlign

import torch.nn.functional as F

class Faster_RCNN(torch.nn.Module):
    def __init__(self,is_train=True):
        super(Faster_RCNN, self).__init__()
        self.is_train=is_train

        self.resNet=resNet()
        self.fpn=PyramidFeatures()
        self.rpn=Rpn(is_train=self.is_train)
        self.roialign_layer=nn.ModuleList([RoIAlign(cfg.roialign_size,1/cfg.fpn_strides[i],sampling_ratio=0) for i in range(len(cfg.roialign_layers))])

        input_channels=cfg.fpn_channels*cfg.roialign_size**2
        self.shared_fc1=nn.Linear(input_channels,cfg.head_base_channels)
        self.shared_fc2=nn.Linear(cfg.head_base_channels,cfg.head_base_channels)
        self.fc_cls=nn.Linear(cfg.head_base_channels,cfg.num_classes+1)
        self.fc_reg=nn.Linear(cfg.head_base_channels,cfg.num_classes*4)

        loss_config = {}
        loss_config["sample_nums"] = cfg.head_nums
        loss_config["pos_fraction"] = cfg.head_pos_fraction
        loss_config["encode_mean"] = cfg.head_encode_mean
        loss_config["encode_std"] = cfg.head_encode_std
        loss_config["num_classes"] = cfg.num_classes
        loss_config["neg_th"] = cfg.head_neg_th
        loss_config["pos_th"] = cfg.head_pos_th
        self.loss=Loss(loss_config,is_rpn=False)

        inference_config = {}
        inference_config["encode_mean"] = cfg.head_encode_mean
        inference_config["encode_std"] = cfg.head_encode_std
        inference_config["num_classes"] = cfg.num_classes
        inference_config["nms_threshold"] = cfg.head_nms_threshold
        inference_config["nms_post"] = cfg.head_nms_post
        inference_config["pos_th"] = cfg.head_pos_th_test
        inference_config["cls_output_channels"] = cfg.num_classes+1
        self.inference = Inference(inference_config, is_rpn=False)

        nn.init.normal_(self.fc_cls.weight,mean=0,std=0.01)
        nn.init.normal_(self.fc_reg.weight,mean=0,std=0.001)
        for m in [self.fc_cls,self.fc_reg]:
            nn.init.constant_(m.bias,0)

        for m in [self.shared_fc1,self.shared_fc2]:
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias,0)

    def collect_proposal(self, proposals):
        batch_valids=[]
        batch_proposals=[]
        batch_size=len(proposals)
        for i in range(len(proposals)):
            proposal_per_img=proposals[i]

            proposal=proposal_per_img["bbox"]
            batch_proposals.append(proposal)
            nums_proposal=proposal.size(0)
            valid_per_img=proposal.new_zeros((batch_size*nums_proposal,),dtype=torch.bool)
            valid_per_img[i*nums_proposal:(i+1)*nums_proposal]=1
            batch_valids.append(valid_per_img)
        batch_proposals=torch.cat(batch_proposals)

        return batch_proposals,batch_valids

    def extract_roi(self,features,roi_location):
        scale=torch.sqrt((roi_location[:,3]-roi_location[:,1])*(roi_location[:,4]-roi_location[:,2]))
        target_lvls=torch.floor(torch.log2(scale/cfg.finest_scale+1e-6))
        target_lvls=target_lvls.clamp(min=0,max=len(features)-2).long()

        roi_feats=features[0].new_zeros(roi_location.size(0),cfg.fpn_channels,cfg.roialign_size,cfg.roialign_size)

        for i in range(len(features)-1):
            mask=target_lvls==i
            inds=mask.nonzero(as_tuple=False).squeeze(1)
            if(inds.numel()>0):
                rois_=roi_location[inds]
                roi_feats_t=self.roialign_layer[i](features[i],rois_)
                roi_feats[inds]=roi_feats_t
            else:
                roi_feats+=sum(x.view(-1)[0] for x in self.parameters())*0. + features[i].sum()*0.

        return roi_feats

    def forward(self,images,ori_img_shape=None,res_img_shape=None,pad_img_shape=None,gt_bboxes=None,gt_labels=None):
        loss={}

        c2,c3,c4,c5=self.resNet(images)
        features=self.fpn([c2,c3,c4,c5])
        rpn_loss, proposals=self.rpn(features,res_img_shape=res_img_shape,pad_img_shape=pad_img_shape,gt_bboxes=gt_bboxes)

        proposals,valids=self.collect_proposal(proposals)
        if(self.is_train):
            targets=self.loss.compute_targets(proposals,valids,gt_bboxes,gt_labels)
            roi_location = []

            for i in range(len(targets)):
                target_per_img = targets[i]
                pos_proposal = target_per_img["pos_proposal"]
                neg_proposal = target_per_img["neg_proposal"]
                proposal_per_img = torch.cat([pos_proposal, neg_proposal], dim=0)
                imgid = proposal_per_img.new_ones((proposal_per_img.size(0), 1)) * i
                proposal_per_img = torch.cat([imgid, proposal_per_img], dim=1)

                roi_location.append(proposal_per_img)

            roi_location = torch.cat(roi_location)
            roi_feats=self.extract_roi(features,roi_location)
            roi_feats=roi_feats.flatten(1)

            roi_feats=self.shared_fc1(roi_feats)
            roi_feats=F.relu(roi_feats,inplace=True)
            roi_feats=self.shared_fc2(roi_feats)
            roi_feats=F.relu(roi_feats,inplace=True)

            x_cls=roi_feats
            x_reg=roi_feats

            cls_pred=self.fc_cls(x_cls)
            reg_pred=self.fc_reg(x_reg)

            head_loss=self.loss(cls_pred,reg_pred,targets)

            loss["rpn_cls_loss"]=rpn_loss["cls_loss"]
            loss["rpn_reg_loss"]=rpn_loss["reg_loss"]
            loss["head_cls_loss"]=head_loss["cls_loss"]
            loss["head_reg_loss"]=head_loss["reg_loss"]

            return loss
        else:
            imd_id=proposals.new_zeros(size=(proposals.size(0),1))
            roi_location=torch.cat([imd_id,proposals],dim=1)
            roi_feats=self.extract_roi(features,roi_location)
            roi_feats = roi_feats.flatten(1)

            roi_feats = self.shared_fc1(roi_feats)
            roi_feats = F.relu(roi_feats, inplace=True)
            roi_feats = self.shared_fc2(roi_feats)
            roi_feats = F.relu(roi_feats, inplace=True)

            x_cls = roi_feats
            x_reg = roi_feats

            cls_pred = self.fc_cls(x_cls)
            reg_pred = self.fc_reg(x_reg)

            scores, bboxes, labels=self.inference(cls_pred,reg_pred,proposals,res_img_shape)

            scale_factor=ori_img_shape.float()/res_img_shape.float()
            scale_factor=torch.cat([scale_factor,scale_factor],dim=1)
            bboxes=bboxes*scale_factor

            return scores, bboxes, labels

