from data import *
from utils.augmentations import SSDAugmentation
from layers.modules import MultiBoxLoss
from ssd import build_ssd
import os
import sys
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
import argparse
import matplotlib.pyplot as plt
import torchvision
import random
import os.path as osp
import matplotlib.pyplot as plt

DATASET_ROOT = "mydata/big_cat"
DATASET_NAME = "/mydata"
cfg = SubT
BASE_NET = "weight/vgg16_reducedfc.pth"
DATA_DETECTION = SUBTDetection
BATCH_SIZE = 4#10
PRETRAINED_MODEL = None
PRETRAINED_ITER = 0
SAVE_MODEL_ITER = 500
START_ITER = 0
NUM_WORKERS = 4
CUDA = True
LR = 1e-4#1e-3
MOMENTUM = 0.4
WEIGHT_DECAY = 5e-3#5e-4
GAMMA = 0.1
VISDOM = False
SAVE_FOLDER = "weight/" + DATASET_NAME +"/"
if not os.path.exists(SAVE_FOLDER):
    os.makedirs(SAVE_FOLDER)
# print('done')
# print(DATA_DETECTION)


if torch.cuda.is_available():
    if not CUDA:
        print("WTF are u wasting your CUDA device?")
    else:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

# Initial model weights & bias
def xavier(param):
    init.xavier_uniform(param)
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()

# Adjust learning rate during training
def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = LR * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        print("Change learning rate to: ", lr)



############################## dataset ##########################
dataset = DATA_DETECTION(root=DATASET_ROOT, image_sets=['train'],transform=SSDAugmentation(cfg['min_dim'], MEANS))

classes = dataset.target_transform.class_to_ind
# print(classes)
# print("Class to index: \n", classes)
classes = sorted(classes.items(), key=lambda kv: kv[1])
label = []
for i in classes:
    label.append(i[0])
label.append('None')



# Delcare SSD Network
ssd_net = build_ssd('train', cfg['min_dim'], cfg['num_classes'])
# ssd_net = build_ssd('train', cfg['min_dim'], 2)

if CUDA:
    net = torch.nn.DataParallel(ssd_net)
    cudnn.benchmark = True

if PRETRAINED_MODEL is not None: # Use SSD pretrained model
    print('Resuming training, loading {}...'.format(PRETRAINED_MODEL))
    ssd_net.load_weights(SAVE_FOLDER + PRETRAINED_MODEL)
else:
    print('Initializing weights...')
    vgg_weights = torch.load(BASE_NET) # load vgg pretrained model
    ssd_net.vgg.load_state_dict(vgg_weights)
    ssd_net.extras.apply(weights_init) # Initial SSD model weights & bias
    ssd_net.loc.apply(weights_init)
    ssd_net.conf.apply(weights_init)


net = ssd_net
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=MOMENTUM,
                weight_decay=WEIGHT_DECAY)
# print(cfg['min_dim'])
criterion = MultiBoxLoss(BATCH_SIZE ,cfg['num_classes'], 0.5, True, 0, True, 3, 0.5,False, CUDA)

loss_list = []


net.train()
# loss counters
loc_loss = 0
conf_loss = 0
epoch = 0
# print('Loading the dataset...')
epoch_size = len(dataset) // BATCH_SIZE
# print('Training SSD on:', DATASET_NAME)

data_loader = data.DataLoader(dataset, BATCH_SIZE,
                num_workers=NUM_WORKERS,
                shuffle=True, collate_fn=detection_collate,
                pin_memory=True)
batch_iterator = iter(data_loader)


step_index = 0
for iteration in range(START_ITER, cfg['max_iter']):
    if iteration in cfg['lr_steps']:
        step_index += 1
        adjust_learning_rate(optimizer, GAMMA, step_index)

    # make sure data iter not out of range
    try:
        images, targets = next(batch_iterator)
        # print(targets[0][0][4].item(), label[int(targets[0][0][4].item())])
    except StopIteration:
        batch_iterator = iter(data_loader)
        images, targets = next(batch_iterator)
    if CUDA:
        images = Variable(images.cuda())
        targets = [Variable(ann.cuda(), volatile=True) for ann in targets]
    else:
        images = Variable(images)
        targets = [Variable(ann, volatile=True) for ann in targets]

    # Forward
    t0 = time.time()
    out = net(images)
    # backprop
    optimizer.zero_grad()
    loss_l, loss_c = criterion(out, targets)
    loss = loss_l + loss_c
    loss.backward()
    optimizer.step()
    t1 = time.time()
    loc_loss += loss_l.item()
    conf_loss += loss_c.item()

    loss_list.append(loss.item())

    if iteration % 10 == 0:
            print('timer: %.4f sec.' % (t1 - t0))
            print('iter ' + repr(PRETRAINED_ITER + iteration) + ' || Loss: %.4f ||' % (loss.item()), end='')
            plt.plot(loss_list)
            plt.xlabel('iteration')
            plt.ylabel('loss')
            plt.title('SSD Traning Loss')
            plt.savefig("loss.png")

    if iteration != 0 and iteration % SAVE_MODEL_ITER == 0:
            print('Saving state, iter:', iteration)
            torch.save(ssd_net.state_dict(), SAVE_FOLDER + DATASET_NAME + "_" +
                       repr(PRETRAINED_ITER + iteration) + '.pth')
# Save final model
torch.save(ssd_net.state_dict(),
            SAVE_FOLDER + DATASET_NAME + '.pth')
