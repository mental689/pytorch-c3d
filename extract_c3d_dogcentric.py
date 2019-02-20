import os, logging, argparse, django, cv2, pickle
import numpy as np
from tqdm import tqdm
from glob import glob
from network.transform import CenterCrop, SubtractMean
from network.c3d import C3D
from torchvision import transforms
from django.conf import settings
import torch
from torch import nn
django.setup()
logger = logging.getLogger(__name__)


def load(video_path, transforms=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error('Cannot open video {}'.format(video_path))
        return None, None
    results = []
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    chunk = np.zeros(shape=(16, 171, 128, 3))
    for i in range(num_frames//16+1):
        for j in range(min(16, num_frames-16*i)):
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            frame = cv2.resize(frame, dsize=(171, 128))
            chunk[i] = frame
            if transforms is not None:
                chunk = transforms(chunk)
            results.append(chunk)
    cap.release()
    return results


def load_model(model_path):
    model = C3D(n_classes=101)
    model.cuda()
    model = nn.DataParallel(model)
    try:
        model.module.load_state_dict(torch.load(model_path))
        model.module.eval()
        model.eval()
    except Exception as e:
        logger.error(e)
    return model


def extract(video_path, model, layer='fc6'):
    trans = transforms.Compose([
        SubtractMean(),
        CenterCrop()
    ])
    segments = load(video_path, trans)
    features = []
    for segment in tqdm(segments):
        features.append(model.extract(torch.from_numpy(segment), layer=layer).float())
    return features


def get_video_list():
    video_dir = '{}/static/videos/robotics-ftp.ait.kyushu-u.ac.jp/dogcentric/'.format(settings.BASE_DIR)
    video_list = glob(video_dir + '**/**/*.avi')
    return video_list


def parse_args():
    p = argparse.ArgumentParser('Extracting C3D features for DogCentric dataset')
    p.add_argument('-layer', type=str, default='fc6')
    p.add_argument('-save_dir', type=str, default='{}/static/features/'.format(settings.BASE_DIR))
    p.add_argument('-model', type=str, help='path to the model weight')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    videos = get_video_list()
    model = load_model(args.model)
    for video in videos:
        logger.info(video)
        features = extract(video_path=video, model=model, layer=args.layer)
        pickle.dump(features, open('{}_features.pkl'.format(video), 'wb'))

