# coding: utf-8
from network.trainer import *
video_train_list = '/media/tuananhn/903a7d3c-0ce5-444b-ad39-384fcda231ed/UCF101/video-caffe/examples/c3d_ucf101/c3d_ucf101_train_split1.txt'
video_test_list = '/media/tuananhn/903a7d3c-0ce5-444b-ad39-384fcda231ed/UCF101/video-caffe/examples/c3d_ucf101/c3d_ucf101_test_split1.txt'
trainer = UCFClipTrainer(video_train_list, video_test_list, batch_size=1)
trainer.load(path='static/models/model_000008.pt')
val_loss, top1, top5 = trainer.test()
print('Epoch 0/10: Top-1 accuracy {:.2f} %, Top-5 accuracy: {:.2f} %'.format(top1.item(), top5.item()))
