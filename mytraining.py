import torch
import torch.nn as nn
import myKit
import warnings
warnings.filterwarnings("ignore")

""""具体训练参数设置"""

# loss = nn.CrossEntropyLoss(reduction="none")
loss_fn = nn.L1Loss(reduction='sum')

# criterion = nn.CrossEntropyLoss(reduction='none')
if __name__ == '__main__':

    # net = myKit.get_net(isEnsemble=False)
    # 6.13 MMCA层直接调用训练好的模块
    net = myKit.get_net(512, 32, depth=6, hidden_dim=256, embedding_dim=1024, decoder_embed_dim=512, gender_size=32)
    # 6.17 直接使用ResNet50来训练
    # net = resnet50(pretrained=True)
    lr = 5e-3
    batch_size = 32
    # num_epochs = 50
    num_epochs = 30
    # MAE_epochs = 50
    MAE_epochs = 30
    weight_decay = 0.0001
    lr_period = 10
    lr_decay = 0.5
    # bone_dir = os.path.join('..', 'data', 'archive', 'testDataset')
    bone_dir = "../archive"
    csv_name = "boneage-training-dataset.csv"
    train_df, valid_df = myKit.split_data(bone_dir, csv_name, 20, 0.1, 256)
    train_set, val_set = myKit.create_data_loader(train_df, valid_df)
    # torch.set_default_tensor_type('torch.FloatTensor')
    myKit.train_fn(net=net, train_dataset=train_set, valid_dataset=val_set, num_epochs=num_epochs, MAE_epochs=MAE_epochs, lr=lr, wd=weight_decay, lr_period=lr_period, lr_decay=lr_decay,loss_fn=loss_fn, batch_size=batch_size, model_path="model_MAE.pth", record_path="RECORD_reg.csv", MAE_record_path="RECORD_MAE.csv")
