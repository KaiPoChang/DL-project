#!/usr/bin/env python
# coding: utf-8

# # Import Library

# In[1]:


import torchvision
import torchvision.models as models
from torchvision import transforms, datasets, utils

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler
from torch.autograd import Variable

import time
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook as tqdm
from skimage import io, transform


# # Import Pre-train Model

# In[2]:


pretrain_resnet18 = models.resnet18(pretrained=True)
pretrain_resnet50 = models.resnet50(pretrained=True)


# In[3]:


# Freezing all layers
all_resnet = [pretrain_resnet18, pretrain_resnet50]
for resnet in all_resnet:
    for param in resnet.parameters():
        param.requires_grad = False

#修改最後一層的output_features為5
pretrain_resnet18.fc = nn.Linear(51200, 5)
pretrain_resnet50.fc = nn.Linear(204800, 5)


# # Construct model

# In[4]:


def conv3x3(inplanes, outplanes, stride=1, groups=1):
    return nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=stride, padding=1, groups=groups, bias=False)


# In[5]:


class Basicblock(nn.Module):
    expansion = 1
    
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Basicblock, self).__init__()

        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
        
        out += residual
        out = self.relu(out)
        
        return out


# In[6]:


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=5):
        self.inplanes = 64
        super(ResNet, self).__init__()     
        
        # Network input
        self.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2),padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * block.expansion, 5)
        
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes*block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes*block.expansion, kernel_size=1, bias=False),
                nn.BatchNorm2d(planes*block.expansion)
                )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
 
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
 
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
 
        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)
 
        return x


# In[7]:


def resnet18(pretrained=False, **kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


# # Data preprocessing

# In[8]:


def getData(mode):
    if mode == 'train':
        train_img = '/home/ym.10612012/Deep_learning_lab/lab3_Diabetic_Retinopathy_Detection/train_img.csv'
        train_label = '/home/ym.10612012/Deep_learning_lab/lab3_Diabetic_Retinopathy_Detection/train_label.csv'
        img = pd.read_csv(train_img)
        label = pd.read_csv(train_label)
        return np.squeeze(img.values), np.squeeze(label.values)
    else:
        test_img = '/home/ym.10612012/Deep_learning_lab/lab3_Diabetic_Retinopathy_Detection/test_img.csv'
        test_label = '/home/ym.10612012/Deep_learning_lab/lab3_Diabetic_Retinopathy_Detection/test_label.csv'
        img = pd.read_csv(test_img)
        label = pd.read_csv(test_label)
        return np.squeeze(img.values), np.squeeze(label.values)


# In[9]:


transformations = transforms.Compose([transforms.ToPILImage(),transforms.RandomHorizontalFlip(0.5), transforms.ToTensor()])


# In[10]:


class RetinopathyLoader(Dataset):
    def __init__(self, root, mode, transform=None):
        
        """
        Args:
            root (string): Root path of the dataset.
            mode : Indicate procedure status(training or testing)

            self.img_name (string list): String list that store all image names.
            self.label (int or float list): Numerical list that store all ground truth label values.
        """
        self.root = root
        self.img_name, self.label = getData(mode)
        self.mode = mode
        self.transform = transform
        print("> Found %d images..." % (len(self.img_name)))

    def __len__(self):
        """'return the size of dataset"""
        return len(self.img_name)

    def __getitem__(self, index):
        """something you should implement here"""
        img_path = self.root + self.img_name[index] + '.jpeg'
        image = io.imread(img_path)

        label = np.array(self.label[index])
        label = torch.from_numpy(label)
        
        if self.transform is not None:
            image = self.transform(image)
            image = image/255
        """
           step1. Get the image path from 'self.img_name' and load it.
                  hint : path = root + self.img_name[index] + '.jpeg'
           
           step2. Get the ground truth label from self.label
                     
           step3. Transform the .jpeg rgb images during the training phase, such as resizing, random flipping, 
                  rotation, cropping, normalization etc. But at the beginning, I suggest you follow the hints. 
                       
                  In the testing phase, if you have a normalization process during the training phase, you only need 
                  to normalize the data. 
                  
                  hints : Convert the pixel value to [0, 1]
                          Transpose the image shape from [H, W, C] to [C, H, W]
                         
            step4. Return processed image and label
        """
        return image, label


# # Training model

# In[11]:


batch_size = 4
num_epochs = 5
learning_rate = 1e-3
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(pretrain_resnet18.parameters(), lr=learning_rate, momentum = 0.9, weight_decay = 5e-4)
# Sets the learning rate of each parameter group to the initial lr decayed by gamma every step_size epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
root = '/home/ym.10612012/Deep_learning_lab/lab3_Diabetic_Retinopathy_Detection/data/data/'


# In[12]:


Retino_train_dataset = RetinopathyLoader(root, mode='train',transform=transformations)
Retino_train_loader = DataLoader(Retino_train_dataset, batch_size=batch_size, shuffle=True)
Retino_train_dataset_sizes = len(Retino_train_dataset)

Retino_test_dataset = RetinopathyLoader(root, mode='test',transform=transformations)
Retino_test_loader = DataLoader(Retino_test_dataset, batch_size=batch_size, shuffle=True)
Retino_test_dataset_sizes = len(Retino_test_dataset)

dataloaders ={'train': Retino_train_loader, 'test': Retino_test_loader}


# In[13]:


def train_model(model):
    since = time.time()
    best_model_wts = model.state_dict()
    best_acc = 0.0
    epoch_acc_train = []
    epoch_acc_test = []
    predict = []
    truth = []
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs-1))
        print('-' * 20)
        for phase in ['train', 'test']:
            if phase == 'train':
                print('Train')
                exp_lr_scheduler.step()
                model.train(True)
            else:
                print('Test')
                model.train(False)

            running_loss = 0.0
            running_corrects = 0

            for img, label in tqdm(dataloaders[phase]):
                batch_img = Variable(img).cuda()
                batch_label = Variable(label).cuda()
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(batch_img)
                if type(outputs) == tuple:
                    outputs, _ = outputs    
                _, preds = torch.max(outputs.data, 1)
                loss = loss_func(outputs, batch_label)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()    

                # statistics
                running_loss += loss.data
                running_corrects += torch.sum(preds == batch_label.data)
                predict.extend(preds)
                truth.extend(batch_label.data)

            if phase == 'train':
                epoch_loss = running_loss.data.cpu().numpy() / Retino_train_dataset_sizes
                epoch_acc = running_corrects.data.cpu().numpy() / Retino_train_dataset_sizes
                epoch_acc = np.around(epoch_acc, decimals=3)
                epoch_acc_train.append(epoch_acc)
                print(epoch_acc_train)
            else:
                epoch_loss = running_loss.data.cpu().numpy() / Retino_test_dataset_sizes
                epoch_acc = running_corrects.data.cpu().numpy() / Retino_test_dataset_sizes
                epoch_acc = np.around(epoch_acc, decimals=3)
                epoch_acc_test.append(epoch_acc)
                print(epoch_acc_test)
            
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
        
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}hr {:.0f}m {:.0f}s'.format(
            time_elapsed // (360*60), time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights&model
        model.load_state_dict(best_model_wts)
    if model == pretrain_resnet18:
        torch.save(model, 'pretrain_resnet18.pth.tar')
    elif model == unpretrain_resnet18:
        torch.save(model, 'unpretrain_resnet18.pth.tar')

    return epoch_acc_train, epoch_acc_test, predict, truth
# return 的predict和truth是[tensor(0), tensor(5),...]組成的list


# # Conduction

# In[14]:


unpretrain_resnet18 = ResNet(Basicblock, [2, 2, 2, 2]).cuda()


# In[15]:


unpretrain_resnet18 = unpretrain_resnet18.cuda()
optimizer = torch.optim.SGD(unpretrain_resnet18.parameters(), lr=learning_rate, momentum = 0.9, weight_decay = 5e-4)
unpretrain_resnet18 = train_model(unpretrain_resnet18)


# In[19]:


pretrain_resnet18 = pretrain_resnet18.cuda()
optimizer = torch.optim.SGD(pretrain_resnet18.parameters(), lr=learning_rate, momentum = 0.9, weight_decay = 5e-4)
pretrain_resnet18 = train_model(pretrain_resnet18)


# In[20]:


from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=True,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    
    title = 'Normalized confusion matrix'
    # Compute confusion matrix
    print(len(y_true), len(y_pred))
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]))

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


# In[22]:


pre18_acc_train, pre18_acc_test, pre18_truth, pre18_predict = pretrain_resnet18
unpre18_acc_train, unpre18_acc_test, unpre18_truth, unpre18_predict = unpretrain_resnet18


# In[23]:


pre18_true = [i.data.cpu().numpy() for i in pre18_truth]
pre18_pred= [i.data.cpu().numpy() for i in pre18_predict]
unpre18_true = [i.data.cpu().numpy() for i in unpre18_truth]
unpre18_pred= [i.data.cpu().numpy() for i in unpre18_predict]


# In[24]:


plot_confusion_matrix(pre18_true, pre18_pred, classes=5, normalize=True)
plt.title('Normalized confusion matrix (Pretrained ResNet18)')
plt.xlabel('True label')
plt.ylabel('Predicted label')
plt.show()


# In[25]:


plot_confusion_matrix(unpre18_true, unpre18_pred, classes=5, normalize=True)
plt.title('Normalized confusion matrix (Unpretrained ResNet18)')
plt.xlabel('True label')
plt.ylabel('Predicted label')
plt.show()


# In[30]:


epoch = np.arange(0, num_epochs, 1)

plt.figure()
plt.plot(epoch, pre18_acc_train, label='Test (w/o pretraining)', color='blue')
plt.plot(epoch, pre18_acc_test, label='Test (with pretraining)', color='red')
plt.plot(epoch, unpre18_acc_train, label='Train (w/o pretraining)', color='yellow')
plt.plot(epoch, unpre18_acc_test, label='Train (with pretraining)', color='black')

plt.title('Result_comprehension (ResNet18)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

plt.legend()
plt.show() 

