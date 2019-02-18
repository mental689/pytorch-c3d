# coding: utf-8
from network.trainer import *

# Please modify the path to the locations of video lists
video_train_list = '/media/tuananhn/903a7d3c-0ce5-444b-ad39-384fcda231ed/UCF101/video-caffe/examples/c3d_ucf101/c3d_ucf101_train_split1.txt'
video_test_list = '/media/tuananhn/903a7d3c-0ce5-444b-ad39-384fcda231ed/UCF101/video-caffe/examples/c3d_ucf101/c3d_ucf101_test_split1.txt'

# Training
trainer = UCFClipTrainer(video_train_list, video_test_list, batch_size=20)
trainer.train(testing=False, lr=0.003)
