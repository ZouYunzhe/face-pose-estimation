import torch
from torch.autograd import Variable
import numpy as np
from torchvision import transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from PIL import Image
from skimage import io
import operator
import matplotlib.pyplot as plt
import scipy.io as sio

# train_dir = "/home/zouyunzhe/JointTracking/tmp/posetrack/posetrack_data/annotations/data_after_filter/nn_data/shoulder_train_50/"
# test_dir = "/home/zouyunzhe/JointTracking/tmp/posetrack/posetrack_data/annotations/data_after_filter/nn_data/shoulder_test_50/"
PIC_NUMBER = 61992
width_last_layer = 100 // 8

IS_TRAIN = True
BATCH_SIZE = 2
LR = 0.001
THRESHHOLD_TRAIN = 0.15
THRESHHOLD_VAL = 0.15
gpu_id = 1

text_train = '/home/guorui/tmp/input/input_' + str(PIC_NUMBER) + '_regression.txt'
text_val = '/home/guorui/tmp/input/input_5622_regression_val.txt'
checkpoint = torch.load('models/my_model109665.pth.tar')
model_save_name = "my_model" + str(PIC_NUMBER) + "_mytest.pth.tar"
mean_face_mat_path = sio.loadmat('/home/guorui/matlab_code/mean_face_' + str(PIC_NUMBER) + '.mat')
mean_face = mean_face_mat_path['mean_face']


# -----------------ready the dataset--------------------------
def default_loader(path):
    # return Image.open(path).convert('RGB')
    return io.imread(path)


class MyDataset(Dataset):
    def __init__(self, txt, transform=None, target_transform=None, loader=default_loader):
        fh = open(txt, 'r')
        imgs = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split(',')
            imgs.append((words[0], np.float(words[1]), np.float(words[2]), np.float(words[3])))
        self.input_w = 224
        self.input_h = 224
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        fn, label1, label2, label3 = self.imgs[index]
        img = self.loader(fn)
        sub_img = img - mean_face
        sub_img = ((sub_img - np.min(sub_img)) / np.max(sub_img - np.min(sub_img))) * 255
        if self.transform is not None:
            img = self.transform(sub_img)
        return fn, img, label1, label2, label3

    def __len__(self):
        return len(self.imgs)


#
# transform_test = transforms.Compose([
#     transforms.Scale((224, 224, 3)),
#     transforms.ToTensor()
# ])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

train_data = MyDataset(txt=text_train, transform=transform_test)
test_data = MyDataset(txt=text_val, transform=transform_test)
train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
test_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, num_workers=2)

##### plot input data ######
binwidth = 0.05
data_size = len(train_data.imgs)
data1 = [train_data.imgs[i][1] for i in range(data_size)]
data2 = [train_data.imgs[i][2] for i in range(data_size)]
data3 = [train_data.imgs[i][3] for i in range(data_size)]
plt.hist(data1, bins=np.arange(min(data1), max(data1) + binwidth, binwidth), label='batch_label1')
plt.hist(data2, bins=np.arange(min(data2), max(data2) + binwidth, binwidth), label='batch_label2')
plt.hist(data3, bins=np.arange(min(data3), max(data3) + binwidth, binwidth), label='batch_label3')
plt.legend(loc='upper left')
plt.show()


# -----------------create the Net and training------------------------

class Net1(torch.nn.Module):
    def __init__(self, num_classes=3, batch_normalization=False):
        super(Net1, self).__init__()
        self.do_bn = batch_normalization
        self.bn_input = nn.BatchNorm1d(1, momentum=0.5)
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),  # conv1_1
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),  # conv1_2
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # pool1

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # conv2_1
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=5, padding=2),  # conv2_2
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # pool2

            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # conv3_1
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # conv3_2
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # conv3_3
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # pool3

            nn.Conv2d(256, 512, kernel_size=3, padding=1),  # # conv4_1
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),  # # conv4_2
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),  # # conv4_3
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # # pool4

            nn.Conv2d(512, 512, kernel_size=3, padding=1),  # # conv5_1
            nn.BatchNorm2d(512, momentum=0.5),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512, momentum=0.5),  # # conv5_2
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512, momentum=0.5),  # # conv5_3
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # # pool5
        )
        self.classifier = nn.Sequential(
            # nn.Dropout(),
            nn.Linear(512 * 7 * 7, 1024),
            nn.BatchNorm1d(1024, momentum=0.5),
            nn.ReLU(inplace=True),
            # nn.Dropout(),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024, momentum=0.5),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 3)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 512 * 7 * 7)
        x = self.classifier(x)

        return x


## load pretrained model

model = Net1()
# model_dict=model.state_dict()

model.load_state_dict(checkpoint)
print(model)

model.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
loss_func = torch.nn.MSELoss()
if IS_TRAIN:
    for epoch in range(1000):
        print('epoch {}'.format(epoch + 1))

        # training-----------------------------
        model.train()
        train_loss = 0.
        train_acc = 0.
        for index, batch_image, batch_label1, batch_label2, batch_label3 in train_loader:
            # print(index)
            batch_image, batch_label1, batch_label2, batch_label3 = Variable(batch_image.cuda()), Variable(
                batch_label1.cuda()), Variable(batch_label2.cuda()), Variable(batch_label3.cuda())
            out = model(batch_image)
            batch_label1 = batch_label1.type(torch.cuda.FloatTensor)
            batch_label2 = batch_label2.type(torch.cuda.FloatTensor)
            batch_label3 = batch_label3.type(torch.cuda.FloatTensor)
            batch_label = torch.stack((batch_label1, batch_label2, batch_label3), dim=1)
            loss = loss_func(out, batch_label)
            train_loss += loss.data[0]
            # pred = torch.max(out, 1)[1]
            # train_correct = (out == batch_label).sum()

            for i in range(batch_image.shape[0]):
                accuracy = torch.add(out[i], -batch_label[i])
                c = [operator.gt(THRESHHOLD_TRAIN, float(i)) for i in torch.abs(accuracy)]
                if (c[0] & c[1] & c[2]) == True:
                    train_acc += 1
            # train_acc = train_acc / float(batch_image.shape[0])
            # print('current train Loss: {:.6f}, Acc: {:.6f}'.format((float(loss.data)), (float(train_acc))))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('Train Loss: {:.6f}, Acc: {:.6f}'.format(train_loss / (len(
            train_data)), train_acc / (len(train_data))))

        # evaluation--------------------------------
        model.eval()
        eval_loss = 0.
        eval_acc = 0.
        for index, batch_image, batch_label1, batch_label2, batch_label3 in test_loader:
            # print(index)
            batch_image, batch_label1, batch_label2, batch_label3 = Variable(batch_image.cuda(),
                                                                             volatile=True), Variable(
                batch_label1.cuda(), volatile=True), Variable(batch_label2.cuda(), volatile=True), Variable(
                batch_label3.cuda(), volatile=True),
            out = model(batch_image)
            batch_label1 = batch_label1.type(torch.cuda.FloatTensor)
            batch_label2 = batch_label2.type(torch.cuda.FloatTensor)
            batch_label3 = batch_label3.type(torch.cuda.FloatTensor)
            batch_label = torch.stack((batch_label1, batch_label2, batch_label3), dim=1)
            loss = loss_func(out, batch_label)
            eval_loss += loss.data[0]
            for i in range(batch_image.shape[0]):
                accuracy = torch.add(out[i], -batch_label[i])
                c = [operator.gt(THRESHHOLD_VAL, float(i)) for i in torch.abs(accuracy)]
                if (c[0] & c[1] & c[2]) == True:
                    eval_acc += 1
        print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(
            test_data)), eval_acc / (len(test_data))))
        print('Finished Training')
        torch.save(model.state_dict(), 'models/' + model_save_name)
else:
    checkpoint = torch.load('models/' + model_save_name)
    model.load_state_dict(checkpoint)
    model.eval()
    eval_loss = 0.
    eval_acc = 0.
    for batch_x, batch_y in test_loader:
        batch_x, batch_y = Variable(batch_x.cuda(), volatile=True), Variable(batch_y.cuda(), volatile=True)
        out = model(batch_x)
        loss = loss_func(out, batch_y)
        eval_loss += loss.data[0]
        pred = torch.max(out, 1)[1]
        num_correct = (pred == batch_y).sum()
        eval_acc += num_correct.data[0]
    print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(
        test_data)), eval_acc / (len(test_data))))
