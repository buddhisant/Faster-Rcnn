import torch
import utils
import math
from torch import nn
import torch.nn.functional as F
import config as cfg

class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()

    def forward(self,logits,labels,avg_factor):
        labels=labels.view(-1,1).float()
        labels=1-labels
        loss=F.binary_cross_entropy_with_logits(logits,labels,reduction="none")
        loss=loss.sum()/avg_factor
        return loss

class CELoss(nn.Module):
    def __init__(self):
        super(CELoss, self).__init__()

    def forward(self,logits,labels,avg_factor):
        loss=F.cross_entropy(logits,labels,reduction="none")
        loss=loss.sum()/avg_factor
        return loss

class L1Loss(nn.Module):
    def __init__(self):
        super(L1Loss, self).__init__()

    def forward(self,preds,targets,avg_factor):
        loss=torch.abs(preds-targets)
        loss=loss.sum()/avg_factor
        return loss

class Loss(nn.Module):
    def __init__(self, config, is_rpn=True):
        super(Loss, self).__init__()
        self.is_rpn=is_rpn
        self.sample_nums=config.get("sample_nums",None)
        self.pos_fraction=config.get("pos_fraction",None)

        self.encode_mean = config.get("encode_mean",None)
        self.encode_std = config.get("encode_std",None)
        self.num_classes = config.get("num_classes",None)

        self.neg_th = config.get("neg_th",None)
        self.pos_th = config.get("pos_th",None)
        self.th_by_gt = config.get("th_by_gt",None)

        self.class_loss = BCELoss() if self.is_rpn else CELoss()
        self.reg_loss=L1Loss()

    def choice_sample(self, inds, nums):
        if(inds.size(0)<=nums):
            return inds
        perm=torch.randperm(inds.size(0),device=inds.device)[:nums]
        rand_inds=inds[perm]
        return rand_inds

    def compute_targets(self,anchors,valids,gt_bboxes,gt_labels=None):
        targets=[]

        for i, valid in enumerate(valids):
            target_per_img={}
            gt_bbox=gt_bboxes[i]
            gt_label=gt_labels[i] if (not gt_labels is None) else None
            valid_anchor=anchors[valid]
            if(not self.is_rpn):
                valid_anchor=torch.cat([gt_bbox,valid_anchor])

            assigned_gt_inds=torch.full((valid_anchor.size(0),),fill_value=-1,device=valid_anchor.device,dtype=torch.long)
            overlaps=utils.compute_iou_xyxy(gt_bbox,valid_anchor)
            max_overlap,argmax_overlap=overlaps.max(dim=0)
            max_gt_overlap,argmax_gt_overlap=overlaps.max(dim=1)

            neg_inds=max_overlap<self.neg_th
            pos_inds=max_overlap>=self.pos_th
            assigned_gt_inds[neg_inds]=0
            assigned_gt_inds[pos_inds]=argmax_overlap[pos_inds]+1

            if(self.th_by_gt is not None):
                for j in range(gt_bbox.size(0)):
                    if(max_gt_overlap[j]>=self.th_by_gt):
                        index=(overlaps[j,:]==max_gt_overlap[j])
                        assigned_gt_inds[index]=j+1

            pos_inds=torch.nonzero(assigned_gt_inds>0,as_tuple=False).squeeze(1)
            neg_inds=torch.nonzero(assigned_gt_inds==0,as_tuple=False).squeeze(1)
            pos_inds=self.choice_sample(pos_inds,self.sample_nums*self.pos_fraction)
            pos_inds=torch.unique(pos_inds)
            neg_inds=self.choice_sample(neg_inds,self.sample_nums-pos_inds.size(0))
            neg_inds=torch.unique(neg_inds)
            pos_anchors=valid_anchor[pos_inds]
            pos_gt_bboxes=gt_bbox[assigned_gt_inds[pos_inds]-1,:]

            if(gt_label is None):
                pos_gt_labels=pos_gt_bboxes.new_zeros((pos_gt_bboxes.size(0),),dtype=torch.long)
            else:
                pos_gt_labels = gt_label[assigned_gt_inds[pos_inds] - 1]
            neg_gt_labels = pos_gt_labels.new_full((neg_inds.size(0),),self.num_classes)
            mean=pos_anchors.new_tensor(self.encode_mean).view(1,4)
            std=pos_anchors.new_tensor(self.encode_std).view(1,4)

            reg_target_per_img=utils.reg_encode(pos_anchors,pos_gt_bboxes,mean,std)
            target_per_img["valid_inds"]=valid
            target_per_img["pos_inds"]=pos_inds
            target_per_img["neg_inds"]=neg_inds
            target_per_img["cls_labels"]=torch.cat([pos_gt_labels,neg_gt_labels],dim=0)
            target_per_img["reg_targets"]=reg_target_per_img
            if(not self.is_rpn):
                target_per_img["pos_proposal"]=pos_anchors
                target_per_img["neg_proposal"]=valid_anchor[neg_inds]
            targets.append(target_per_img)
        return targets


    def dense_pred_collect(self,cls_preds,reg_preds, targets):
        cls_preds_batch = []
        reg_preds_batch = []
        for i in range(cls_preds[0].size(0)):
            target_per_img=targets[i]
            cls_preds_per_img=[cls_pred[i,:].permute(1,2,0).reshape(-1,self.num_classes) for cls_pred in cls_preds]
            reg_preds_per_img=[reg_pred[i,:].permute(1,2,0).reshape(-1,self.num_classes*4) for reg_pred in reg_preds]
            cls_preds_per_img=torch.cat(cls_preds_per_img,dim=0)
            reg_preds_per_img=torch.cat(reg_preds_per_img,dim=0)

            valid_flag_per_img=target_per_img["valid_inds"]
            cls_preds_per_img=cls_preds_per_img[valid_flag_per_img,:]
            reg_preds_per_img=reg_preds_per_img[valid_flag_per_img,:]

            pos_inds_per_img=target_per_img["pos_inds"]
            neg_inds_per_img=target_per_img["neg_inds"]
            pos_cls_preds_per_img=cls_preds_per_img[pos_inds_per_img,:]
            neg_cls_preds_per_img=cls_preds_per_img[neg_inds_per_img,:]
            cls_preds_batch.append(torch.cat([pos_cls_preds_per_img,neg_cls_preds_per_img],dim=0))
            reg_preds_batch.append(reg_preds_per_img[pos_inds_per_img,:])

        cls_preds_batch=torch.cat(cls_preds_batch,dim=0)
        reg_preds_batch=torch.cat(reg_preds_batch,dim=0)
        return cls_preds_batch, reg_preds_batch

    def sparse_pred_collect(self, cls_preds, reg_preds, targets):
        reg_preds=reg_preds.reshape(reg_preds.size(0),-1,4)

        reg_preds_batch=[]
        for i in range(len(targets)):
            target_per_img=targets[i]
            cls_labels_per_img=target_per_img["cls_labels"]

            pos_inds_per_img=(cls_labels_per_img>=0)&(cls_labels_per_img<self.num_classes)
            pos_inds_per_img=torch.nonzero(pos_inds_per_img,as_tuple=False).squeeze(1)

            pos_labels=cls_labels_per_img[pos_inds_per_img]

            reg_preds_per_img=reg_preds[i*self.sample_nums:(i+1)*self.sample_nums]
            pos_reg_preds_per_img=reg_preds_per_img[pos_inds_per_img,pos_labels]
            reg_preds_batch.append(pos_reg_preds_per_img)

        reg_preds_batch=torch.cat(reg_preds_batch)
        cls_preds_batch=cls_preds

        return cls_preds_batch,reg_preds_batch


    def forward(self, cls_preds, reg_preds, targets):
        cls_targets_batch=[]
        reg_targets_batch=[]
        for i in range(len(targets)):
            target_per_img=targets[i]

            cls_labels=target_per_img["cls_labels"]
            cls_targets_batch.append(cls_labels)
            reg_targets_batch.append(target_per_img["reg_targets"])

        cls_targets_batch=torch.cat(cls_targets_batch,dim=0)
        reg_targets_batch=torch.cat(reg_targets_batch,dim=0)
        if(self.is_rpn):
            cls_preds_batch, reg_preds_batch = self.dense_pred_collect(cls_preds, reg_preds, targets)
        else:
            cls_preds_batch, reg_preds_batch = self.sparse_pred_collect(cls_preds,reg_preds, targets)

        total_samples=cls_preds_batch.size(0)
        cls_loss=self.class_loss(cls_preds_batch,cls_targets_batch,total_samples)
        reg_loss=self.reg_loss(reg_preds_batch,reg_targets_batch,total_samples)

        return {"cls_loss":cls_loss,"reg_loss":reg_loss}