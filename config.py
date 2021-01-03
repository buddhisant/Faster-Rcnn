# 训练集标注文件的路径
train_ann_path = "/home/buddhisant/data/coco/annotations/instances_val2017.json"
# 训练集图片文件夹的路径
train_img_path = "/home/buddhisant/data/coco/images/val2017"
# 训练集标注文件的路径
val_ann_path = "/home/buddhisant/data/coco/annotations/instances_val2017.json"
# 训练集图片文件夹的路径
val_img_path = "/home/buddhisant/data/coco/images/val2017"
# 保存checkpoint的文件夹的路径
archive_path = "./archive"
# 保存测试结果的文件夹的路径
output_path = "./output"

# 对图片进行resize的尺度范围
resize_scale = (1333, 800)

# 随机水平翻转的概率
flip_pro = 0.5

# 对图片进行normalize的均值
norm_mean = [123.675, 116.28, 103.53]
# 对图片进行normalize的标准差
norm_std = [58.395, 57.12, 57.375]
# 是否将图片转化为rgb格式
norm_to_rgb = True

# 每个gpu上的图片数量
samples_per_gpu = 2

# 每个gpu上的worker数量
num_workers_per_gpu = 0

# 采用的resnet的深度，取值范围为[18,34,50,101,152]
resnet_depth = 50

# 需要固定参数的resnet的layer数，取值范围为[0,1,2,3,4]
freeze_stages = 1

# backbone网络layer1输出的channel数量
c2_channels = 256
# backbone网络layer2输出的channel数量
c3_channels = 512
# backbone网络layer3输出的channel数量
c4_channels = 1024
# backbone网络layer4输出的channel数量
c5_channels = 2048
# fpn网络输出的channel数量
fpn_channels = 256
# fpn网络的stride
fpn_strides = [4, 8, 16, 32, 64]

#对每一层生成anchor时的anchor相对大小
anchor_scale = 8
#每一组anchor的长宽比
anchor_ratio = [2,1,0.5]

# 数据集中类别的数量
num_classes = 80

#在训练开始的前num_warmup_iters次迭代里，采取warmup操作
num_warmup_iters=500
#采用constant的warmup操作
warmup_factor=0.001
#lr衰减率
lr_decay_factor=0.1
#lr衰减的时间点
lr_decay_time=[9, 12]
#训练的最大epoch数量
max_epochs=12

#基础学习率
base_lr=0.02
#基础weight_decay率
weight_decay=0.0001
#优化器的动量
momentum=0.9

#在decode时，对预测得到的dw和dh，限定其范围，即限定预测框相对于真实框的比例
decode_ratio_clip=0.016

#rpn网络在训练时负样本的阈值
rpn_neg_th=0.3
#rpn网络在训练时正样本的阈值
rpn_pos_th=0.7
#rpn网络在训练时由gt制定最佳正样本时的阈值
rpn_gt_pos_th=0.3
#rpn网络在训练时，样本总数
rpn_nums=256
#rpn网络在训练时，正样本所占比例
rpn_pos_fraction=0.5
#rpn网络回归分支的target, 编码时的mean
rpn_encode_mean=[0,0,0,0]
#rpn网络回归分支的target, std
rpn_encode_std=[1.0,1.0,1.0,1.0]
#训练过程中，rpn生成proposal时，在每一层参加nms的最大数量
rpn_nms_pre=2000
#rpn生成proposal时，nms采用的阈值
rpn_nms_threshold=0.7
#rpn最多保留的proposal的数量
rpn_nms_post=1000

#测试过程中，rpn生成proposal时，在每一层参加nms的最大数量
rpn_nms_pre_test=1000

#head网络在训练时，样本总数
head_nums=512
#head网络在训练时，正样本所占比例
head_pos_fraction=0.25
#head网络回归分支的target, 编码时的mean
head_encode_mean=[0,0,0,0]
#head网络回归分支的target, std
head_encode_std=[0.1,0.1,0.2,0.2]
#inference时，nms之后保留的最大预测框数量
head_nms_post=100
#inference时，判断一个预测框是不是候选集的阈值，默认采用0.05
head_pos_th_test=0.05
#head网络生成预测框时，nms采用的阈值
head_nms_threshold=0.5
#head网络隐藏层的channels数量
head_base_channels=1024
#head部分的网络在划分负样本时的iou阈值
head_neg_th=0.5
#head部分的网络在划分正样本时的iou阈值
head_pos_th=0.5

#只对fpn网络的前4层应用roialign操作
roialign_layers=[0,1,2,3]
#roialign操作输出的矩形框的长宽
roialign_size=7

finest_scale=56

#保存checkpoint文件时的前缀名
check_prefix="faster_rcnn"