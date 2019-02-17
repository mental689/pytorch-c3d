C3D by PyTorch
=====

## Introduction
We reproduce the results of C3D [1] in UCF101 [2] and test the performance in DogCentric Activity dataset [3].

## Preparation

### Dataset

* To download UCF101 please visit [http://crcv.ucf.edu/data/UCF101.php](http://crcv.ucf.edu/data/UCF101.php) to download a rar archive.
Please extract the archive and modify the [train](https://github.com/chuckcho/video-caffe/blob/master/examples/c3d_ucf101/c3d_ucf101_train_split1.txt) and [test](https://github.com/chuckcho/video-caffe/blob/master/examples/c3d_ucf101/c3d_ucf101_test_split1.txt) lists of `video-caffe` to match the video paths.

* To download the [DogCentric Activity dataset](http://robotics.ait.kyushu-u.ac.jp/yumi/db/first_dog.html), please run the script [`static/videos/download_dogcentric.sh`](./static/videos/download_dogcentric.sh) in the directory `static/videos`.

```bash
$ cd static/videos
$ bash download_dogcentric.sh
```

## References

[1] D. Tran, L. Bourdev, R. Fergus, L. Torresani, and M. Paluri, "Learning Spatiotemporal Features with 3D Convolutional Networks", ICCV 2015.

[2] K. Soomro, A.-R. Zamir and M. Shah, "UCF101: A Dataset of 101 Human Action Classes From Videos in The Wild", CRCV-TR-12-01, November, 2012. 

[3] Y. Iwashita, A. Takamine, R. Kurazume, and M. S. Ryoo, "First-Person Animal Activity Recognition from Egocentric Videos", ICPR 2014. 