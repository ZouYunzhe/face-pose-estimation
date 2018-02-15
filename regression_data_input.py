import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from skimage import io
import scipy.io as sio
import os
from tensorboardX import SummaryWriter
from datetime import datetime

PIC_NUMBER = 61992
BATCH_SIZE = 20
LR = 0.00001
THRESHHOLD_TRAIN = 0.2
THRESHHOLD_VAL = 0.2
gpu_id = 3
Need_Pretrained_model = True

time = datetime.now().strftime('%m-%d_%H_%M')
Tb_path = os.path.join('log/Resnet50', time)
if not os.path.isdir(Tb_path):
    os.makedirs(Tb_path)

text_train = '/home/guorui/tmp/input/input_' + str(PIC_NUMBER) + '_regression.txt'
text_train_class = '/home/guorui/tmp/input/input_' + str(PIC_NUMBER) + '_classification_torch.txt'
# text_train_class = '/home/guorui/tmp/input/input_' + str(PIC_NUMBER) + '_classification_torch.txt'
# text_train_regress = '/home/guorui/tmp/input/input_' + str(PIC_NUMBER) + '_regression_torch.txt'
text_val = '/home/guorui/tmp/input/input_11337_classification_torch.txt'
text_test = '/home/guorui/tmp/input/input_BIWI_regression_test.txt'

mean_face_mat_path_train = sio.loadmat('/home/guorui/matlab_code/mean_face_61992.mat')
mean_face_train = mean_face_mat_path_train['mean_face']

mean_face_mat_path_val = sio.loadmat('/home/guorui/matlab_code/input_8670_regression_val.mat')
mean_face_val = mean_face_mat_path_val['mean_face']

mean_face_mat_path_test = sio.loadmat('/home/guorui/tmp/input/mean_face_BIWI.mat')
mean_face_test = mean_face_mat_path_test['mean_face']

model_save_name = "my_model" + str(PIC_NUMBER) + '_' + str(LR) + "_Pretrained_ResNet.pth.tar"
checkpoint_path = 'models/my_model61992_0.001_Pretrained_ResNet.pth.tar'

writer = SummaryWriter(Tb_path)


# -----------------ready the dataset--------------------------
def default_loader(path):
    # return Image.open(path).convert('RGB')
    return io.imread(path)


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),

])


class MyDataset(Dataset):
    def __init__(self, is_train=False, is_val=False, is_test=False, transforms=transform, target_transform=None,
                 loader=default_loader):
        self.is_train = is_train
        self.is_val = is_val
        self.is_test = is_test

        if self.is_train:
            fh = open(text_train, 'r')
        elif self.is_val:
            fh = open(text_test, 'r')
        else:
            fh = open(text_val, 'r')

        imgs = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split(',')
            imgs.append((words[0], np.float(words[1]), np.float(words[2]), np.float(words[3])))
        self.input_w = 224
        self.input_h = 224
        self.imgs = imgs
        self.transform = transforms
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        fn, label1, label2, label3 = self.imgs[index]
        img = self.loader(fn)
        if self.is_train == True:
            sub_img = img - mean_face_train
        elif self.is_val == True:
            sub_img = img - mean_face_test
        else:
            sub_img = img - mean_face_val
        sub_img = ((sub_img - np.min(sub_img)) / np.max(sub_img - np.min(sub_img))) * 255

        if self.transform is not None:
            img = self.transform(sub_img)
        return fn, img, label1, label2, label3

    def __len__(self):
        return len(self.imgs)


class VGG(Dataset):
    def __init__(self, is_train=False, is_val=False, is_test=False, transforms=transform, target_transform=None,
                 loader=default_loader):
        self.is_train = is_train
        self.is_val = is_val
        self.is_test = is_test

        if self.is_train:
            fh = open(text_train, 'r')
        elif self.is_val:
            fh = open(text_test, 'r')
        else:
            fh = open(text_val, 'r')

        imgs = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split(',')
            imgs.append((words[0], np.float(words[1]), np.float(words[2]), np.float(words[3])))
        self.input_w = 224
        self.input_h = 224
        self.imgs = imgs
        self.transform = transforms
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        fn, yaw, pitch, roll = self.imgs[index]
        img = self.loader(fn)

        yaw = np.rad2deg(yaw)
        roll = np.rad2deg(roll)
        pitch = np.rad2deg(pitch)

        bins = np.array(range(-180, 180, 2))
        binned_pose = np.digitize([yaw, pitch, roll], bins) - 1
        labels = torch.LongTensor(binned_pose)
        cont_labels = torch.FloatTensor([yaw, pitch, roll])

        if self.transform is not None:
            img = self.transform(img)
        return fn, img, labels, cont_labels

    def __len__(self):
        return len(self.imgs)


class VGG_2(Dataset):
    def __init__(self,PIC_NUMBER, is_train=False, is_val=False, is_test=False, transforms=transform, target_transform=None,
                 loader=default_loader):
        self.is_train = is_train
        self.is_val = is_val
        self.is_test = is_test

        if self.is_train:
            fh = open(text_train_class, 'r')
        elif self.is_val:
            fh = open(text_val, 'r')
        else:
            fh = open(text_val, 'r')

        imgs = []
        imgs_class = []

        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split(',')
            imgs_class.append((words[0], np.float(words[1]), np.float(words[2]), np.float(words[3]), np.float(words[4]),
                               np.float(words[5]), np.float(words[6])))
        # for line in fh_regression:
        #     line = line.strip('\n')
        #     line = line.rstrip()
        #     words = line.split(',')
        #     imgs_regress.append((words[0], np.float(words[1]), np.float(words[2]), np.float(words[3])))


        self.input_w = 224
        self.input_h = 224
        self.imgs = imgs
        # self.imags_regress = imgs_regress
        self.imags_class = imgs_class
        self.transform = transforms
        self.target_transform = target_transform
        self.loader = loader

        # fn_class, yaw_class, pitch_class, roll_class = self.imags_class[1]
        # fn_regress, yaw_regress, pitch_regress, roll_regress = self.imags_regress[1]
        #
        # img_class = self.loader(fn_class)
        # labels_class = torch.FloatTensor([yaw_class, pitch_class, roll_class])
        # labels_regress = torch.FloatTensor([yaw_regress, pitch_regress, roll_regress])
        #
        # if self.transform is not None:
        #     img_class = self.transform(img_class)

    def __getitem__(self, index):

        fn_class, yaw_class, pitch_class, roll_class, yaw_regress, pitch_regress, roll_regress = self.imags_class[index]
        # fn_regress, yaw_regress, pitch_regress, roll_regress = self.imags_regress[index]

        img_class = self.loader(fn_class)
        labels_class = torch.FloatTensor([yaw_class, pitch_class, roll_class])
        labels_regress = torch.FloatTensor([yaw_regress, pitch_regress, roll_regress])

        if self.transform is not None:
            img_class = self.transform(img_class)
        return fn_class, img_class, labels_class, labels_regress

    def __len__(self):
        return len(self.imags_class)

        #
        ###### plot input data ######
        # binwidth = 0.05
        # data_size = len(train_data.imgs)
        # data1 = [train_data.imgs[i][1] for i in range(data_size)]
        # data2 = [train_data.imgs[i][2] for i in range(data_size)]
        # data3 = [train_data.imgs[i][3] for i in range(data_size)]
        # plt.hist(data1, bins=np.arange(min(data1), max(data1) + binwidth, binwidth), label='batch_label1')
        # plt.hist(data2, bins=np.arange(min(data2), max(data2) + binwidth, binwidth), label='batch_label2')
        # plt.hist(data3, bins=np.arange(min(data3), max(data3) + binwidth, binwidth), label='batch_label3')
        # plt.legend(loc='upper left')
        # plt.show()
