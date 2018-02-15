import torch
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.autograd import Variable
import os
from tensorboardX import SummaryWriter
from datetime import datetime
from regression_data_input import MyDataset
from my_regress_class_tools import train_processing
from regression_tools import testing_processing
from regression_data_input import VGG, VGG_2
from new_vet import Hopenet
import operator
from torch.optim import lr_scheduler

NET = 'ResNet50'
PIC_NUMBER = 61992
IS_TRAIN = True
BATCH_SIZE = 50
LR = 0.000001
THRESHHOLD_TRAIN = 20
THRESHHOLD_VAL = 20
gpu_id = 4
Need_Pretrained_model = True
weight_decay = 0
alpha = 0.001
num_epochs = 100
train_acc = 0

time = datetime.now().strftime('%m-%d_%H_%M')
Tb_path = os.path.join('log/' + NET, time)
if not os.path.isdir(Tb_path):
    os.makedirs(Tb_path)

model_save_name = 'models/my_model2_' + NET + '_' + str(PIC_NUMBER) + '_' + str(alpha) + '_' + str(LR) + '_' + str(
    BATCH_SIZE) + '_Class.pth.tar'

result_mat_path = '/home/guorui/tmp/output/' + 'my_model_' + NET + '_' + str(PIC_NUMBER) + '_' + str(
    LR) + '_Regression_v3.mat'
checkpoint_path = 'models/my_model2_ResNet50_61992_0.01_0.0001_80_Class.pth.tar'

writer = SummaryWriter(Tb_path)

# -----------------ready the dataset--------------------------
with torch.cuda.device(gpu_id):
    train_data = VGG_2(PIC_NUMBER, is_train=True)
    validation_data = VGG_2(PIC_NUMBER, is_val=True)
    # test_data = VGG_2(PIC_NUMBER,is_test=True)
    train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
    validation_loader = DataLoader(dataset=validation_data, batch_size=BATCH_SIZE)
    # test_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE)

    # -----------------create the Net and training------------------------
    model = Hopenet(models.resnet.Bottleneck, [3, 4, 6, 3], 180)

    # -----------------build model and load pretrained parameters------------------------
    if Need_Pretrained_model:
        print('loading pretrained model from pytorch')
        ## load pretrained parameters
        Resnet50 = models.resnet50(pretrained=True)
        pretrained_dict = Resnet50.state_dict()
        model_dict = model.state_dict()

        # 1 filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2 overwrite entires in the existing data dict
        model_dict.update(pretrained_dict)
        # 3 load the new state dict
        model.load_state_dict(model_dict)
        # model.load_state_dict(checkpoint)
    else:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint)
    print(model)
    model.cuda()

    criterion = nn.CrossEntropyLoss()
    # Regression loss coefficient
    softmax = nn.Softmax()
    reg_criterion = nn.MSELoss()

    idx_tensor = [idx for idx in range(180)]
    idx_tensor = Variable(torch.cuda.FloatTensor(idx_tensor))

    # optimizer = torch.optim.Adam([{'params': get_ignored_params(model), 'lr': 0},
    #                               {'params': get_non_ignored_params(model), 'lr': LR},
    #                               {'params': get_fc_params(model), 'lr': LR * 5}],
    #                              lr=LR)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=weight_decay)
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    # --------------------------training-------------------------------
    len_train_data=len(train_data)
    len_val_data=len(validation_data)
    if IS_TRAIN:
        train_processing(model, train_loader, num_epochs, criterion, reg_criterion, softmax, idx_tensor, alpha,
                         THRESHHOLD_VAL, optimizer, model_save_name, validation_loader, gpu_id, writer, BATCH_SIZE,
                         len_train_data, len_val_data, None)
    # ---------------------------testing--------------------------------
    else:
        testing_processing(checkpoint_path, model, test_loader, loss_func, test_data, THRESHHOLD_TRAIN, result_mat_path)
