import torch
import config as cfg
import utils

import torch.nn.functional as F

class Inference():
    def __init__(self,config,is_rpn=True):
        # self.is_rpn表示当前是rpn网络还是head网络，由于rpn网络是密集检测，而head网络是稀疏网络，因此需要不同的处理。
        self.is_rpn=is_rpn
        self.encode_mean = config.get("encode_mean",None)
        self.encode_std = config.get("encode_std",None)
        self.num_classes = config.get("num_classes",None)
        self.nms_pre = config.get("nms_pre",None)
        self.nms_threshold = config.get("nms_threshold",None)
        self.nms_post = config.get("nms_post",None)
        self.pos_th=config.get("pos_th",None)
        self.cls_output_channels = config.get("cls_output_channels",None)

    def danse_inference(self, cls_preds, reg_preds, anchors, res_img_shapes):
        """
        为rpn网络进行inference计算
        """
        batch_size = len(res_img_shapes)

        results = []
        for i in range(batch_size):
            # 注意到这里对于采用了detach操作，是为了保证head网络部分的loss不会对rpn网络造成影响
            cls_preds_per_img = [cls_pred[i, :].permute(1, 2, 0).reshape(-1, self.cls_output_channels).detach() for
                                 cls_pred in cls_preds]
            reg_preds_per_img = [reg_pred[i, :].permute(1, 2, 0).reshape(-1, self.num_classes * 4).detach() for reg_pred
                                 in reg_preds]

            num_levels = len(cls_preds_per_img)

            candidate_scores = []
            candidate_anchors = []
            candidate_factors = []
            candidate_ids = []
            for j in range(num_levels):
                anchors_per_level = anchors[j]
                cls_preds_per_level = cls_preds_per_img[j]
                cls_preds_per_level = cls_preds_per_level.reshape(-1).sigmoid()
                reg_preds_per_level = reg_preds_per_img[j]

                if (cls_preds_per_level.size(0) > self.nms_pre):
                    ranked_scores, rank_inds = cls_preds_per_level.sort(descending=True)
                    topk_inds = rank_inds[:self.nms_pre]
                    cls_preds_per_level = ranked_scores[:self.nms_pre]
                    anchors_per_level = anchors_per_level[topk_inds, :]
                    reg_preds_per_level = reg_preds_per_level[topk_inds, :]

                ids = cls_preds_per_level.new_ones((anchors_per_level.size(0),), dtype=torch.long) * j

                candidate_scores.append(cls_preds_per_level)
                candidate_anchors.append(anchors_per_level)
                candidate_factors.append(reg_preds_per_level)
                candidate_ids.append(ids)

            candidate_scores = torch.cat(candidate_scores, dim=0)
            candidate_factors = torch.cat(candidate_factors, dim=0)
            candidate_anchors = torch.cat(candidate_anchors, dim=0)
            candidate_ids = torch.cat(candidate_ids, dim=0)

            mean = candidate_anchors.new_tensor(self.encode_mean).view(1, 4)
            std = candidate_anchors.new_tensor(self.encode_std).view(1, 4)
            proposal = utils.reg_decode(candidate_anchors, candidate_factors, mean, std, res_img_shapes[i])
        
            scores, bboxes = utils.ml_nms(candidate_scores, proposal, candidate_ids, self.nms_threshold)
            scores = scores[:self.nms_post]
            bboxes = bboxes[:self.nms_post]

            results.append({"score": scores, "bbox": bboxes})

        return results

    def sparse_inference(self, cls_preds, reg_preds, anchors, res_img_shapes):
        """
        为head网络进行inference计算
        """
        cls_scores=F.softmax(cls_preds,dim=1)
        cls_scores=cls_scores[:,:-1].contiguous()
        #首先根据预测出的分类得分，初步筛选出正样本
        mask_pos=cls_scores>=self.pos_th

        pos_location=torch.nonzero(mask_pos,as_tuple=False)
        pos_inds=pos_location[:,0]
        pos_labels=pos_location[:,1]

        reg_preds=reg_preds.view(reg_preds.size(0),-1,4)

        scores = cls_scores[pos_inds,pos_labels]
        anchors=anchors[pos_inds,:]
        reg_preds = reg_preds[pos_inds,pos_labels,:]

        mean = anchors.new_tensor(self.encode_mean).view(1, 4)
        std = anchors.new_tensor(self.encode_std).view(1, 4)
        proposal = utils.reg_decode(anchors, reg_preds, mean, std, res_img_shapes[0])

        scores,bboxes,labels=utils.mc_nms(scores,proposal,pos_labels,self.nms_threshold)
        scores=scores[:self.nms_post]
        bboxes=bboxes[:self.nms_post]
        labels=labels[:self.nms_post]

        return scores, bboxes, labels

    def __call__(self, cls_preds,reg_preds,anchors,res_img_shapes):
        if(self.is_rpn):
            return self.danse_inference(cls_preds,reg_preds,anchors,res_img_shapes)
        else:
            return self.sparse_inference(cls_preds,reg_preds,anchors,res_img_shapes)
