import datetime
import logging
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.optim import SGD
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from torch.utils.data.dataloader import default_collate

logger = logging.getLogger(__name__)

from network.c3d import C3D
from network.transform import RandomCrop, RandomHorizontalFlip, SubtractMean
from data.ucf101 import UCF101


def collate(batch):
    "Puts each data field into a tensor with outer dimension batch size"
    batch = list(filter(lambda x: x[0] is not None, batch))
    return default_collate(batch)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class UCFClipTrainer(object):
    def __init__(self, train_video_list, test_video_list, use_gpu=True, batch_size=30, new_width=171, new_height=128,
                 crop_size=112, length=16):
        # Preparing data
        train_dataset = UCF101(video_list=train_video_list, subset='train', length=length, new_width=new_width,
                               new_height=new_height,
                               transforms=transforms.Compose([
                                   SubtractMean([102, 98, 90]),
                                   RandomCrop(crop_size),
                                   RandomHorizontalFlip()
                               ]))
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate)
        test_dataset = UCF101(video_list=test_video_list, subset='test', length=length, new_width=new_width,
                              new_height=new_height,
                              transforms=transforms.Compose([
                                  SubtractMean([102, 98, 90]),
                                  RandomCrop(crop_size)
                              ]))
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate)
        self.datasets = (train_dataset, test_dataset)
        self.loaders = (train_loader, test_loader)

        # Preparing network
        self.model = C3D(n_classes=101)
        if use_gpu:
            self.model.cuda()
        self.model = nn.DataParallel(self.model)
        # Print some logs
        conf = {
            'train_video_list': os.path.abspath(train_video_list),
            'test_video_list': os.path.abspath(test_video_list),
            'use_gpu': use_gpu,
            'batch_size': batch_size,
            'resize': (new_width, new_height),
            'crop': crop_size,
            'clip_length': length,
            'mean': [90, 98, 102]
        }
        self.conf = conf
        logger.info('Setup a trainer with parameters:')
        logger.info('{}'.format(self.conf))

    def train(self, epochs=10, lr=0.003, save_model=True, save_dir='./static/models', testing=True):
        # Setup optimizers
        # IT is a little bit complicated, but to match caffe implementation, it must be like this.
        optimizer = SGD([
            {'params': self.model.module.conv1.weight},
            {'params': self.model.module.conv1.bias, 'lr': 2 * lr, 'weight_decay': 0.0},
            {'params': self.model.module.conv2.weight},
            {'params': self.model.module.conv2.bias, 'lr': 2 * lr, 'weight_decay': 0.0},
            {'params': self.model.module.conv3a.weight},
            {'params': self.model.module.conv3a.bias, 'lr': 2 * lr, 'weight_decay': 0.0},
            {'params': self.model.module.conv3b.weight},
            {'params': self.model.module.conv3b.bias, 'lr': 2 * lr, 'weight_decay': 0.0},
            {'params': self.model.module.conv4a.weight},
            {'params': self.model.module.conv4a.bias, 'lr': 2 * lr, 'weight_decay': 0.0},
            {'params': self.model.module.conv4b.weight},
            {'params': self.model.module.conv4b.bias, 'lr': 2 * lr, 'weight_decay': 0.0},
            {'params': self.model.module.conv5a.weight},
            {'params': self.model.module.conv5a.bias, 'lr': 2 * lr, 'weight_decay': 0.0},
            {'params': self.model.module.conv5b.weight},
            {'params': self.model.module.conv5b.bias, 'lr': 2 * lr, 'weight_decay': 0.0},
            {'params': self.model.module.fc6.weight},
            {'params': self.model.module.fc6.bias, 'lr': 2 * lr, 'weight_decay': 0.0},
            {'params': self.model.module.fc7.weight},
            {'params': self.model.module.fc7.bias, 'lr': 2 * lr, 'weight_decay': 0.0},
            {'params': self.model.module.fc8.weight},
            {'params': self.model.module.fc8.bias, 'lr': 2 * lr, 'weight_decay': 0.0},
        ], lr=lr, momentum=0.9, weight_decay=0.005)
        # summary
        log_dir = 'log/{}'.format(datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S'))
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        summary = SummaryWriter(log_dir=log_dir, comment='Training started at {}'.format(datetime.datetime.now()))
        # training
        train_steps_per_epoch = len(self.loaders[0])
        lr_sched = lr_scheduler.StepLR(optimizer, step_size=4 * train_steps_per_epoch, gamma=0.1)
        for i in range(epochs):
            # train
            self.model.train(True)
            categorical_loss = 0.0
            num_iter = 0
            optimizer.zero_grad()
            pbar = tqdm(self.loaders[0])
            for data in pbar:
                num_iter += 1
                # get the inputs
                inputs, labels = data
                # wrap them in Variable
                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda())
                # Compute outputs
                outputs = self.model(inputs)
                # compute loss
                loss = nn.CrossEntropyLoss()(outputs, labels)
                categorical_loss = loss.detach().item()  # /steps_per_update
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                lr_sched.step()
                pbar.set_description(
                    'Epoch {}/{}, Iter {}, Loss: {:.8f}'.format(i + 1, epochs, num_iter, categorical_loss))
                summary.add_scalars('Train loss', {'Loss': categorical_loss},
                                    global_step=i * train_steps_per_epoch + num_iter)

                if num_iter == train_steps_per_epoch:
                    if save_model:
                        if not os.path.exists(save_dir):
                            os.makedirs(save_dir)
                        torch.save(self.model.module.state_dict(), '{}/model_{:06d}.pt'.format(save_dir, i + 1))
                    if testing:
                        val_loss, top1, top5 = self.test()
                        summary.add_scalars('Validation performance', {
                            'Validation loss': val_loss,
                            'Top-1 accuracy': top1,
                            'Top-5 accuracy': top5,
                        }, i)
                        print('Epoch {}/{}: Top-1 accuracy {:.2f} %, Top-5 accuracy: {:.2f} %'.format(i + 1, epochs,
                                                                                                      top1.item(),
                                                                                                      top5.item()))
                    break

    def test(self):
        self.model.train(False)
        categorical_loss = 0.0
        num_iter = 0
        pbar = tqdm(self.loaders[1])
        topk_acc = [0.0, 0.0]
        for data in pbar:
            num_iter += 1
            # get the inputs
            inputs, labels = data
            # wrap them in Variable
            inputs = Variable(inputs.cuda())
            labels = Variable(labels.cuda())
            # Compute outputs
            outputs = self.model(inputs)
            # compute loss
            loss = nn.CrossEntropyLoss()(outputs, labels)
            categorical_loss += loss.detach().item()
            categorical_loss /= num_iter
            topk_acc_ = accuracy(outputs, labels, topk=(1, 5,))
            topk_acc[0] += topk_acc_[0]
            topk_acc[1] += topk_acc_[1]
        return categorical_loss, topk_acc[0] / num_iter, topk_acc[1] / num_iter

    def load(self, path):
        try:
            self.model.module.load_state_dict(state_dict=torch.load(path))
            self.model.module.eval()
            self.model.eval()
        except Exception as e:
            print('Could not load {}'.format(path))
            logger.error(e)
        print('Loaded {}'.format(path))
