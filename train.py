import os
import time
import utils
import solver
import torch
import argparse
import dataset
import transform
import config as cfg
import torch.distributed as dist

from faster_rcnn import Faster_RCNN
from sampler import distributedGroupSampler,groupSampler
from dataloader import build_dataloader

pretrained_path={
    50:"./pretrained/resnet50_pytorch.pth",
    101:"./pretrained/resnet101_pytorch.pth"
}

def train(is_dist,start_epoch,local_rank):
    transforms=transform.build_transforms()
    coco_dataset = dataset.COCODataset(is_train=True, transforms=transforms)
    if(is_dist):
        sampler = distributedGroupSampler(coco_dataset)
    else:
        sampler = groupSampler(coco_dataset)
    dataloader = build_dataloader(coco_dataset, sampler)

    batch_time_meter = utils.AverageMeter()
    rpn_cls_loss_meter = utils.AverageMeter()
    rpn_reg_loss_meter = utils.AverageMeter()
    head_cls_loss_meter = utils.AverageMeter()
    head_reg_loss_meter = utils.AverageMeter()
    losses_meter = utils.AverageMeter()

    model = Faster_RCNN(is_train=True)
    if(start_epoch==1):
        model.resNet.load_pretrained(pretrained_path[cfg.resnet_depth])
    else:
        utils.load_model(model,start_epoch-1)
    model = model.cuda()

    if is_dist:
        model=torch.nn.parallel.DistributedDataParallel(model,device_ids=[local_rank,],output_device=local_rank,broadcast_buffers=False)
    optimizer=solver.build_optimizer(model)
    scheduler=solver.scheduler(optimizer)

    model.train()
    logs=[]

    for epoch in range(start_epoch, cfg.max_epochs + 1):
        if is_dist:
            dataloader.sampler.set_epoch(epoch-1)
        scheduler.lr_decay(epoch)

        end_time = time.time()
        for iteration, datas in enumerate(dataloader, 1):
            scheduler.linear_warmup(epoch,iteration-1)
            images = datas["images"]
            bboxes = datas["bboxes"]
            labels = datas["labels"]
            res_img_shape = datas["res_img_shape"]
            pad_img_shape = datas["pad_img_shape"]

            images = images.cuda()
            bboxes = [bbox.cuda() for bbox in bboxes]
            labels = [label.cuda() for label in labels]

            loss_dict = model(images, gt_bboxes=bboxes, gt_labels=labels,res_img_shape=res_img_shape,pad_img_shape=pad_img_shape)
            rpn_cls_loss = loss_dict["rpn_cls_loss"]
            rpn_reg_loss = loss_dict["rpn_reg_loss"]
            head_cls_loss = loss_dict["head_cls_loss"]
            head_reg_loss = loss_dict["head_reg_loss"]

            losses = rpn_cls_loss + rpn_reg_loss + head_cls_loss + head_reg_loss
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            batch_time_meter.update(time.time()-end_time)
            end_time = time.time()

            rpn_cls_loss_meter.update(rpn_cls_loss.item())
            rpn_reg_loss_meter.update(rpn_reg_loss.item())
            head_cls_loss_meter.update(head_cls_loss.item())
            head_reg_loss_meter.update(head_reg_loss.item())
            losses_meter.update(losses.item())

            if(iteration % 50 == 0):
                if(local_rank == 0):
                    res = "\t".join([
                        "Epoch: [%d/%d]" % (epoch,cfg.max_epochs),
                        "Iter: [%d/%d]" % (iteration, len(dataloader)),
                        "Time: %.3f (%.3f)" % (batch_time_meter.val, batch_time_meter.avg),
                        "rpn_Cls_loss: %.4f (%.4f)" % (rpn_cls_loss_meter.val, rpn_cls_loss_meter.avg),
                        "rpn_Reg_loss: %.4f (%.4f)" % (rpn_reg_loss_meter.val, rpn_reg_loss_meter.avg),
                        "head_Cls_loss: %.4f (%.4f)" % (head_cls_loss_meter.val, head_cls_loss_meter.avg),
                        "head_Reg_loss: %.4f (%.4f)" % (head_reg_loss_meter.val, head_reg_loss_meter.avg),
                        "Loss: %.4f (%.4f)" % (losses_meter.val, losses_meter.avg),
                        "lr: %.6f" % (optimizer.param_groups[0]["lr"]),
                    ])
                    print(res)
                    logs.append(res)
                batch_time_meter.reset()
                rpn_cls_loss_meter.reset()
                rpn_reg_loss_meter.reset()
                head_cls_loss_meter.reset()
                head_reg_loss_meter.reset()
                losses_meter.reset()
        if(local_rank==0):
            utils.save_model(model, epoch)
        if(is_dist):
            utils.synchronize()

    if(local_rank==0):
        with open("logs.txt","w") as f:
            for i in logs:
                f.write(i+"\n")

def main():
    parser=argparse.ArgumentParser(description="Faster RCNN")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--start_epoch", type=int, default=1)
    parser.add_argument("--dist",action="store_true",default=True)


    args=parser.parse_args()
    if(args.dist):
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        utils.synchronize()

    train(args.dist, args.start_epoch, args.local_rank)

if __name__=="__main__":
    main()
