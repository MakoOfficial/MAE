import pandas as pd
from PIL import Image, ImageOps
import os
import numpy as np
import torch
from torch import Tensor
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms
import cv2
from albumentations.augmentations.transforms import Lambda, RandomBrightnessContrast
from albumentations.augmentations.geometric.transforms import ShiftScaleRotate, HorizontalFlip
from albumentations.pytorch.transforms import ToTensorV2
from albumentations.augmentations.crops.transforms import RandomResizedCrop
from albumentations import Compose
import torch.nn as nn
import torch.utils.data as Data
import torch.utils.data.dataset as Dataset
import Animator
from d2l import torch as d2l
import csv
import time
from sklearn.model_selection import train_test_split
import random
from MAE import MAE, Ensemble
from einops import rearrange

"""本文档主要是解决训练过程中的一些所用到的函数的集合
函数列表：
获取神经网络：get_net
标准化数组:standardization
标准化每个通道:sample_normalize
训练集的数据增广:training_compose
"""

# train_df = pd.read_csv('../data/archive/testDataset/train-dataset.csv')
# boneage_mean = train_df['boneage'].mean()
# boneage_div = train_df['boneage'].std()
def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_net(image_size, patch_size, depth=6, hidden_dim=256, embedding_dim=1024, decoder_embed_dim=512, gender_size=32):
    """获取神经网络"""
    # mae = MAE(image_size, patch_size, depth=depth, hidden_dim=hidden_dim, embedding_dim=embedding_dim, decoder_embed_dim=decoder_embed_dim)
    return MAE(image_size, patch_size, depth=depth, hidden_dim=hidden_dim, embedding_dim=embedding_dim, decoder_embed_dim=decoder_embed_dim)
    # return Ensemble(mae, embedding_dim, gender_size)

def sample_normalize(image, **kwargs):
    """标准化"""
    image = image / 255
    mean, std = image.mean(axis=0), image.std(axis=0)
    return (image - mean) / (std + 1.e-6)**.5


# 随机删除一个图片上的像素，p为执行概率，scale擦除部分占据图片比例的范围，ratio擦除部分的长宽比范围
randomErasing = transforms.RandomErasing(scale=(0.02, 0.08), ratio=(0.5, 2), p=0.8)


def randomErase(image, **kwargs):
    """随机删除一个图片上的像素"""
    return randomErasing(image)


transform_train = Compose([
    # 训练集的数据增广
    # 随机大小裁剪，512为调整后的图片大小，（0.5,1.0）为scale剪切的占比范围，概率p为0.5
    # RandomResizedCrop(512, 512, (0.5, 1.0), p=0.5),
    # ShiftScaleRotate操作：仿射变换，shift为平移，scale为缩放比率，rotate为旋转角度范围，border_mode用于外推法的标记，value即为padding_value，前者用到的，p为概率
    ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=20, border_mode=cv2.BORDER_CONSTANT, value=0.0,
                     p=0.8),
    # 水平翻转
    HorizontalFlip(p=0.5),
    # 概率调整图片的对比度
    RandomBrightnessContrast(p=0.8, contrast_limit=(-0.3, 0.2)),
    # 标准化
    Lambda(image=sample_normalize),
    # 将图片转化为tensor类型
    ToTensorV2(),
    # 做随机擦除
    Lambda(image=randomErase)
])

transform_valid = Compose([
    # 验证集的数据处理
    Lambda(image=sample_normalize),
    ToTensorV2()
])


def read_image(file_path, image_size=512):
    """读取图片，并统一修改为512x512"""
    img = Image.open(file_path)
    # 开始修改尺寸
    w, h = img.size
    long = max(w, h)
    # 按比例缩放成512
    w, h = int(w / long * image_size), int(h / long * image_size)
    # 压缩并插值
    img = img.resize((w, h), Image.ANTIALIAS)
    # 然后是给短边扩充，使用ImageOps.expand
    delta_w, delta_h = image_size - w, image_size - h 
    padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
    # 转化成np.array
    return np.array(ImageOps.expand(img, padding).convert('L'))

def one_hot(x):
    """"encode the label to one-hot label"""
    label = torch.zeros(240)
    label[int(x)] = 1

    return label

def split_data(data_dir, csv_name, category_num, split_ratio, aug_num):
    """重构数据级结构"""
    age_df = pd.read_csv(os.path.join(data_dir, csv_name))
    age_df['path'] = age_df['id'].map(lambda x: os.path.join(data_dir,
                                                            csv_name.split('.')[0],
                                                            '{}.png'.format(x)))
    age_df['exists'] = age_df['path'].map(os.path.exists)
    print(age_df['exists'].sum(), 'images found of', age_df.shape[0], 'total')
    age_df['male'] = age_df['male'].astype('float32')
    age_df['gender'] = age_df['male'].map(lambda x:'male' if x else 'female')

    # disable the selected data-category
    # age_df['Bin'] = pd.cut(age_df['boneage'], category_num, labels=False)
    # lower_bound = age_df['Bin'].min() + 5
    # upper_bound = age_df['Bin'].max() - 5
    # selected_df = age_df[age_df['Bin'].between(lower_bound, upper_bound)]
    # global boneage_mean
    # boneage_mean = selected_df['boneage'].mean()
    # global boneage_div
    # boneage_div = selected_df['boneage'].std()
    # selected_df['zscore'] = selected_df['boneage'].map(lambda x: (x-boneage_mean)/boneage_div)
    # selected_df.dropna(inplace = True)
    # selected_df['boneage_category'] = pd.cut(age_df['boneage'], int(category_num-10))

    global boneage_mean
    boneage_mean = age_df['boneage'].mean()
    global boneage_div
    boneage_div = age_df['boneage'].std()
    # we don't want normalization for now
    # boneage_mean = 0
    # boneage_div = 1.0
    age_df['zscore'] = age_df['boneage'].map(lambda x: (x-boneage_mean)/boneage_div)
    age_df.dropna(inplace = True)
    age_df['boneage_category'] = pd.cut(age_df['boneage'], category_num)

    raw_train_df, valid_df = train_test_split(
    age_df,
    test_size=split_ratio,
    random_state=2023,
    stratify=age_df['boneage_category']
    )
    print('train', raw_train_df.shape[0], 'validation', valid_df.shape[0])
    train_df = raw_train_df.groupby(['boneage_category']).apply(lambda x: x.sample(aug_num*2, replace=True)).reset_index(drop=True)
    # 注意的是，这里对df进行多列分组，因为boneage_category为10类， male为2类，所以总共有20类，而apply对每一类进行随机采样，并且有放回的抽取，所以会生成1w的数据
    # train_df = raw_train_df.groupby(['boneage_category']).apply(lambda x: x)
    print('New Data Size:', train_df.shape[0], 'Old Size:', raw_train_df.shape[0])
    # raw_train_df.to_csv("train.csv")
    train_df.to_csv("train.csv")
    valid_df.to_csv("valid.csv")
    # return raw_train_df, valid_df
    return train_df, valid_df
    # return train_df, valid_df, boneage_mean, boneage_div

def soften_labels(l, x):
    "soften the label distribution"
    a = torch.arange(0,240)
    a = 1 - torch.abs(a - x)/l
    relu = nn.ReLU()
    a = relu(a)
    return a

# create 'dataset's subclass,we can read a picture when we need in training trough this way
class BAATrainDataset(Dataset.Dataset):
    """重写Dataset类"""

    def __init__(self, df) -> None:
        self.df = df

    def __getitem__(self, index):
        row = self.df.iloc[index]
        num = int(row['id'])
        # return (transform_train(image=read_iamge(self.file_path, f"{num}.png"))['image'], Tensor([row['male']])), row['boneage']
        # return (transform_train(image=read_image(row["path"]))['image'], Tensor([row['male']])), row[
            # 'zscore']
        return (transform_train(image=read_image(row["path"]))['image'], Tensor([row['male']])), one_hot(row["boneage"])

    def __len__(self):
        return len(self.df)

class BAAValDataset(Dataset.Dataset):

    def __init__(self, df) -> None:
        self.df = df


    def __getitem__(self, index):
        row = self.df.iloc[index]
        num = int(row['id'])
        return (transform_valid(image=read_image(row["path"]))['image'], Tensor([row['male']])), row[
            'boneage']

    def __len__(self):
        return len(self.df)

def create_data_loader(train_df, val_df):
    return BAATrainDataset(train_df), BAAValDataset(val_df)

# criterion = nn.CrossEntropyLoss(reduction='none')
# penalty function
# def L1_penalty(net, alpha):
#     loss = 0
#     for param in net.MLP.parameters():
#         loss += torch.sum(torch.abs(param))

#     return alpha * loss

def try_gpu(i=0):
    """如果存在，则返回gpu(i)，否则返回cpu()"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def train_fn(net, train_dataset, valid_dataset, num_epochs, MAE_epochs, lr, wd, lr_period, lr_decay, loss_fn, batch_size=32, model_path="./model.pth", record_path="./RECORD.csv", MAE_record_path="./RECORD.csv"):
    """将训练函数和验证函数杂糅在一起的垃圾函数"""
    # record outputs of every epoch
    record = [['epoch', 'training loss', 'val loss', 'lr']]
    with open(record_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in record:
            writer.writerow(row)
    MAE_record = [['epoch', 'training loss', 'val loss', 'lr']]
    with open(MAE_record_path, 'w', newline='') as csvfile1:
        writer1 = csv.writer(csvfile1)
        for row in MAE_record:
            writer1.writerow(row)
    devices = torch.device('cuda')
    # 增加多卡训练
    ## Network, optimizer, and loss function creation
    net = net.to(devices)
    
    # 数据读取
    # Creates dataloaders, which load data in batches
    # Note: test loader is not shuffled or sampled
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=6,
        drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=batch_size,
        drop_last=True,
        num_workers=6,
        shuffle=False)


    MSE_fn =  nn.MSELoss(reduction="sum")
    # loss_fn = nn.L1Loss(reduction='sum')
    # loss_fn = nn.BCELoss(reduction="sum")
    lr = lr

    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    # 每过10轮，学习率降低一半
    scheduler = StepLR(optimizer, step_size=lr_period, gamma=lr_decay)

    seed=101
    torch.manual_seed(seed)  

    ## MAE Trains
    
    for epoch in range(MAE_epochs):
        net.train()
        # net.fine_tune(False)
        print(f"MAE TRAINING!!\nEpoch:{epoch+1}")
        MAE_record = []
        global MAE_training_loss
        MAE_training_loss = torch.tensor([0], dtype=torch.float32)
        global MAE_total_size
        MAE_total_size = torch.tensor([0], dtype=torch.float32)

        # xm.rendezvous("initialization")

        start_time = time.time()
        # 在不同的设备上运行该模型

        #   enumerate()，为括号中序列构建索引
        for batch_idx, data in enumerate(train_loader):
            # #put data to GPU
            image, gender = data[0]
            image, gender = torch.squeeze(image.type(torch.FloatTensor).to(devices)), gender.type(torch.FloatTensor).to(devices)

            batch_size = len(data[1])
            label = data[1].to(devices)
            optimizer.zero_grad()

            # _, pred_pic, mask, _ = net(image, gender, 0.75)
            pred_pic, mask, _ = net(image, 0.75)
            target = rearrange(image, 'b (h p1) (w p2) -> b (h w) (p1 p2)', p1 = 32, p2 = 32)
            # mean = target.mean(dim=-1, keepdim=True)
            # var = target.var(dim=-1, keepdim=True)
            # target = (target - mean) / (var + 1.e-6)**.5
            loss = MSE_fn(pred_pic, target)
            loss.backward()
            # backward,update parameter，更新参数
            # 6_3 增大batchsize，若累计8个batch_size更新梯度，或者batch为最后一个batch
            # if (batch_idx + 1) % 8 == 0 or batch_idx == 377 :
            optimizer.step()
            # print("=============更新之后===========")
            # for name, parms in net.named_parameters():	
            #     print('-->name:', name)
            #     print('-->grad_requirs:',parms.requires_grad)
            #     print('-->grad_value:',parms.grad)
            #     print("======================================")
            batch_loss = loss.item()

            MAE_training_loss += batch_loss
            MAE_total_size += batch_size
            # print('epoch', epoch+1, '; ', batch_idx+1,' batch loss:', batch_loss / batch_size)

        ## Evaluation
        # Sets net to eval and no grad context
        MAE_val_total_size, MAE_mae_loss = MAE_valid_fn(net=net, val_loader=val_loader, devices=devices)
        # accuracy_num = accuracy(pred_list[1:, :], grand_age[1:])
        
        MAE_train_loss, MAE_val_mae = MAE_training_loss / MAE_total_size, MAE_mae_loss / MAE_val_total_size
        MAE_record.append([epoch, round(MAE_train_loss.item(), 2), round(MAE_val_mae.item(), 2), optimizer.param_groups[0]["lr"]])
        print(
            f'training loss is {round(MAE_train_loss.item(), 2)}, val loss is {round(MAE_val_mae.item(), 2)}, time : {round((time.time() - start_time), 2)}, lr:{optimizer.param_groups[0]["lr"]}')
        scheduler.step()
        with open(MAE_record_path, 'a+', newline='') as csvfile1:
            writer1 = csv.writer(csvfile1)
            for row in MAE_record:
                writer1.writerow(row)
    
    
    # regression train
    """======================================================================"""
    # optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    # # 每过10轮，学习率降低一半
    # scheduler = StepLR(optimizer, step_size=lr_period, gamma=lr_decay)
#     for epoch in range(num_epochs):
#         net.train()
#         net.fine_tune(True)
#         print(epoch)
#         this_record = []
#         global training_loss
#         training_loss = torch.tensor([0], dtype=torch.float32)
#         global total_size
#         total_size = torch.tensor([0], dtype=torch.float32)

#         # xm.rendezvous("initialization")

#         start_time = time.time()
#         # 在不同的设备上运行该模型

#         #   enumerate()，为括号中序列构建索引
#         for batch_idx, data in enumerate(train_loader):
#             # #put data to GPU
#             image, gender = data[0]
#             image, gender = torch.squeeze(image.type(torch.FloatTensor).to(devices)), gender.type(torch.FloatTensor).to(devices)

#             batch_size = len(data[1])
#             label = data[1].to(devices)

#             # zero the parameter gradients，是参数梯度归0
#             optimizer.zero_grad()
#             _, _, _, y_pred = net(image, gender, 0.75)
#             # y_pred = y_pred.squeeze()

#             # print(y_pred, label)，求损失
#             loss = loss_fn(y_pred, label)
#             # loss = criterion(y_pred, label.long()).sum()
#             # backward,calculate gradients，反馈计算梯度
#             # 弃用罚函数
# #             total_loss = loss + L1_penalty(net, 1e-5)
# #             total_loss.backward() 
#             loss.backward()
#             # backward,update parameter，更新参数
#             # 6_3 增大batchsize，若累计8个batch_size更新梯度，或者batch为最后一个batch
#             # if (batch_idx + 1) % 8 == 0 or batch_idx == 377 :
#             optimizer.step()
#             # print("=============更新之后===========")
#             # for name, parms in net.named_parameters():	
#             #     print('-->name:', name)
#             #     print('-->grad_requirs:',parms.requires_grad)
#             #     print('-->grad_value:',parms.grad)
#             #     print("======================================")
#             batch_loss = loss.item()

#             training_loss += batch_loss
#             total_size += batch_size
#             # print('epoch', epoch+1, '; ', batch_idx+1,' batch loss:', batch_loss / batch_size)

#         ## Evaluation
#         # Sets net to eval and no grad context
#         val_total_size, mae_loss = valid_fn(net=net, val_loader=val_loader, devices=devices)
#         # accuracy_num = accuracy(pred_list[1:, :], grand_age[1:])
        
#         train_loss, val_mae = training_loss / total_size, mae_loss / val_total_size
#         this_record.append([epoch, round(train_loss.item(), 2), round(val_mae.item(), 2), optimizer.param_groups[0]["lr"]])
#         print(
#             f'training loss is {round(train_loss.item(), 2)}, val loss is {round(val_mae.item(), 2)}, time : {round((time.time() - start_time), 2)}, lr:{optimizer.param_groups[0]["lr"]}')
#         scheduler.step()
#         with open(record_path, 'a+', newline='') as csvfile:
#             writer = csv.writer(csvfile)
#             for row in this_record:
#                 writer.writerow(row)
    """======================================================================"""
    torch.save(net, model_path)


def valid_fn(*, net, val_loader, devices):
    """验证函数：输入参数：网络，验证数据，验证性别，验证标签
    输出：返回MAE损失"""
    net.eval()
    global val_total_size
    val_total_size = torch.tensor([0], dtype=torch.float32)
    global mae_loss
    mae_loss = torch.tensor([0], dtype=torch.float32)
    loss_fn = nn.L1Loss(reduction='sum')
    with torch.no_grad():
        for batch_idx, data in enumerate(val_loader):
            val_total_size += len(data[1])

            image, gender = data[0]
            image, gender = torch.squeeze(image.type(torch.FloatTensor).to(devices)), gender.type(torch.FloatTensor).to(devices)

            label = data[1].type(torch.FloatTensor).to(devices)

            #   net内求出的是normalize后的数据，这里应该是是其还原，而不是直接net（）
            _, _, _, y_pred = net(image, gender, 0.75)
            y_pred = y_pred.cpu()
            label = label.cpu()
            # y_pred = y_pred * boneage_div + boneage_mean
            y_pred = y_pred.argmax(axis=1)
            y_pred = y_pred.squeeze()

            batch_loss = loss_fn(y_pred, label).item()
            mae_loss += batch_loss
    return val_total_size, mae_loss

def MAE_valid_fn(*, net, val_loader, devices):
    """验证函数：输入参数：网络，验证数据，验证性别，验证标签
    输出：返回MAE损失"""
    net.eval()
    global MAE_val_total_size
    MAE_val_total_size = torch.tensor([0], dtype=torch.float32)
    global MAE_mae_loss
    MAE_mae_loss = torch.tensor([0], dtype=torch.float32)
    MSE_fn = nn.MSELoss()
    with torch.no_grad():
        for batch_idx, data in enumerate(val_loader):
            MAE_val_total_size += len(data[1])
            
            image, gender = data[0]
            image, gender = torch.squeeze(image.type(torch.FloatTensor).to(devices)), gender.type(torch.FloatTensor).to(devices)

            #   net内求出的是normalize后的数据，这里应该是是其还原，而不是直接net（）
            # _, pred_pic, mask, _ = net(image, gender, 0.75)
            pred_pic, mask, _ = net(image, 0.75)
            target = rearrange(image, 'b (h p1) (w p2) -> b (h w) (p1 p2)', p1 = 32, p2 = 32)
            # mean = target.mean(dim=-1, keepdim=True)
            # var = target.var(dim=-1, keepdim=True)
            # target = (target - mean) / (var + 1.e-6)**.5
            loss = MSE_fn(pred_pic, target)
            # loss = ((loss * mask).sum() / mask.sum())
    
            batch_loss = loss.item()
            MAE_mae_loss += batch_loss
    return MAE_val_total_size, MAE_mae_loss

def loss_map(class_loss, class_num, path):
    """"输入参数：各个年龄的损失class_loss，各个年龄的数量class_num，画出每个年龄的误差图"""
    data = torch.zeros((230, 1))
    for i in range(class_loss.shape[0]):
        if class_num[i]:
            data[i] = class_loss[i] / class_num[i]
    legend = ['MAE']
    animator = Animator.Animator(xlabel='month', xlim=[1, 230], legend=legend)
    for i in range(data.shape[0]):
        animator.add(i, data[i])
    animator.save(path)


if __name__ == '__main__':
    # num_epochs, learning_rate, weight_decay = 10, 2e-4, 5e-4
    # lr_period, lr_decay = 10, 0.5
    # x = torch.rand((16, 3, 512, 512))
    # gender = torch.ones((16, 1))
    # age = torch.ones((16, 1))
    # MMANet = get_net(isEnsemble=True)
    # MMANet = MMANet.to(device=try_gpu())
    # print(sum(p.numel() for p in MMANet.parameters()))
    # params = list(MMANet.MLP.parameters())
    # print(params)

    # bone_dir = "F:\GitCode\BoneAgeAss-main\data/archive/testDataset"
    # csv_name = "boneage-traning-dataset.csv"
    # train_df, valid_df = split_data(bone_dir, csv_name, 10, 0.1, 10)
    # print("boneage_mean = ", boneage_mean, "boneage_div", boneage_div)

    x = read_image('E:\code/archive/boneage-training-dataset/1435.png')
    x = transform_train(image=x)
    print(x)
