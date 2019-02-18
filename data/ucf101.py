import torch
import torch.utils.data as data_utils
import logging
import numpy as np
import cv2
from tqdm import tqdm

logger = logging.getLogger(__name__)


class UCF101(data_utils.Dataset):
    def __init__(self, video_list, subset='train', transforms=None, length=1, new_width=171, new_height=128):
        super(UCF101, self).__init__()
        self.subset = subset
        with open(video_list) as f:
            lines = f.readlines()
        f.close()
        self.videos = []
        for line in tqdm(lines):
            video_file, start_frame_num, class_idx = line.strip().split()
            self.videos.append({
                'video': video_file,
                'start': int(start_frame_num),
                'class': int(class_idx)
            })
        self.transforms = transforms
        self.length = length
        self.new_width = new_width
        self.new_height = new_height

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, item):
        info = self.videos[item]
        cap = cv2.VideoCapture('{}.avi'.format(info['video']))
        if not cap.isOpened():
            logger.error('Cannot open video {}'.format(info['video']))
            return None, None
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, info['start'] - 2)
        chunk = []
        for i in range(info['start'], min(info['start']+self.length, num_frames)):
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            frame = cv2.resize(frame, dsize=(self.new_width, self.new_height))
            chunk.append(frame)
        for i in range(len(chunk), self.length):
            chunk.append(frame)
        try:
            chunk = np.asarray(chunk, dtype=np.float32)
        except:
            return None, None
        if self.transforms is not None:
            chunk = self.transforms(chunk)
        labels = np.asarray([info['class']], dtype=np.int64)
        return torch.from_numpy(chunk.transpose([3,0,1,2])), torch.tensor(info['class'], dtype=torch.int64)


