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
        
        # 构建roialign层，注意默认情况下fpn有5层feature map，但是只对前4层feature map采用roialign操作
        self.roialign_layer=nn.ModuleList([RoIAlign(cfg.roialign_size,1/cfg.fpn_strides[i],sampling_ratio=0) for i in range(len(cfg.roialign_layers))])
        
        # 在head网络中进行检测时，一个roi最终会由一个向量来表示，这里的input_channels表示这个向量的长度
        input_channels=cfg.fpn_channels*cfg.roialign_size**2
        self.shared_fc1=nn.Linear(input_channels,cfg.head_base_channels)
        self.shared_fc2=nn.Linear(cfg.head_base_channels,cfg.head_base_channels)
        self.fc_cls=nn.Linear(cfg.head_base_channels,cfg.num_classes+1)
        self.fc_reg=nn.Linear(cfg.head_base_channels,cfg.num_classes*4)

        loss_config = {} 
        loss_config["sample_nums"] = cfg.head_nums #表示head网络在训练时，每张图片中选取的样本（正样本+负样本）的数量
        loss_config["pos_fraction"] = cfg.head_pos_fraction #在全部样本中，正样本的占比
        loss_config["encode_mean"] = cfg.head_encode_mean 
        loss_config["encode_std"] = cfg.head_encode_std
        loss_config["num_classes"] = cfg.num_classes
        loss_config["neg_th"] = cfg.head_neg_th #head网络在训练时，生成负样本的iou阈值
        loss_config["pos_th"] = cfg.head_pos_th #head网络在训练时，生成正样本的iou阈值
        self.loss=Loss(loss_config,is_rpn=False)

        inference_config = {}
        inference_config["encode_mean"] = cfg.head_encode_mean
        inference_config["encode_std"] = cfg.head_encode_std
        inference_config["num_classes"] = cfg.num_classes
        inference_config["nms_threshold"] = cfg.head_nms_threshold #生成预测框之后，采用nms去冗余的iou阈值
        inference_config["nms_post"] = cfg.head_nms_post #用nms对预测框去冗余之后，只保留一定数量的预测框
        inference_config["pos_th"] = cfg.head_pos_th_test #在nms之前，判断一个预测框是不是正样本，进行初步的判断
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
        """
        在训练head网络时，首先将一个batch中所有图片的proposal集中起来
        :param proposals: rpn为每张图片生成的proposal
        """
        batch_valids=[]
        batch_proposals=[]
        batch_size=len(proposals)
        for i in range(len(proposals)):
            proposal_per_img=proposals[i]

            proposal=proposal_per_img["bbox"]
            batch_proposals.append(proposal)
            nums_proposal=proposal.size(0)
            valid_per_img=proposal.new_zeros((batch_size*nums_proposal,),dtype=torch.bool)
            valid_per_img[i*nums_proposal:(i+1)*nums_proposal]=1 # 因为把一个batch的proposal都集中到了一起，所以就需要为每张图片单独标记出它自己的proposal。
            batch_valids.append(valid_per_img)
        batch_proposals=torch.cat(batch_proposals)

        return batch_proposals,batch_valids

    def extract_roi(self,features,roi_location):
        """
        roialign层
        :param features: 金字塔特征层
        :param roi_location: shape为[N*512, 5]，即roi对应的坐标，其中第一列表示这个roi属于那一张图片，如果属于这个batch的第一张图片，则取值为0。后4列为坐标
        """
        # 需要首相根据每个roi的尺寸，划分到不同层次的feature map上，提取相应的特征
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
        """
        :param ori_img_shape: 图片原始的尺寸
        :param res_img_shape: 默认按照(800,1333)对图片进行放缩，res_img_shape即表示放缩后的尺寸
        :param pad_img_shape: 放缩后的图片尺寸不能保证被32整除，因此需要pad，pad_img_shape即表示pad后的尺寸
        """
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

