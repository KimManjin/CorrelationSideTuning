import torch
import torch.utils.data as data
import decord
import os
import numpy as np
from numpy.random import randint
import io
import pandas as pd
import random
from PIL import Image
import math
import copy


class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return os.path.join(self._data[0].split('_')[0], self._data[0])

    @property
    def start_idx(self):
        return int(self._data[1])

    @property
    def num_frames(self):
        return int(self._data[2]) - int(self._data[1]) + 1

    @property
    def verb_label(self):
        return int(self._data[3])
    
    @property
    def noun_label(self):
        return int(self._data[4])


class Video_dataset(data.Dataset):
    def __init__(self, root_path, list_file, labels_file,
                 num_segments=1, modality='RGB', new_length=1,
                 image_tmpl='img_{:010d}.jpg', transform=None,
                 random_shift=True, test_mode=False,
                 dense_sample=False, test_clips=3,
                 num_sample=1):

        self.root_path = root_path
        self.list_file = list_file
        self.num_segments = num_segments
        self.modality = modality
        self.seg_length = new_length
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode
        self.loop=False
        self.labels_file = labels_file
        self.sample_range = 128
        self.dense_sample = dense_sample  # using dense sample as I3D
        self.test_clips = test_clips
        self.num_sample = num_sample
        if self.dense_sample:
            print('=> Using dense sample for the dataset...')
        if self.num_sample > 1:
            print('=> Using repeated augmentation...')
        self._parse_list()
        self.initialized = False

    @property
    def total_length(self):
        return self.num_segments * self.seg_length
    
    @property
    def classes(self):
        assert isinstance(self.labels_file, list), "labels_file must be a list for epic-kitchens-100"
        if len(self.labels_file) == 1:
            classes_all = pd.read_csv(self.labels_file[0])
            return classes_all.values.tolist()
        elif len(self.labels_file) == 2:
            verb_classes = pd.read_csv(self.labels_file[0])
            noun_classes = pd.read_csv(self.labels_file[1]) 
            return [verb_classes.values.tolist(), noun_classes.values.tolist()]
        else:
            raise ValueError("labels_file list must have length 1 or 2")
    
    def _parse_list(self):
        # check the frame number is large >3:
        tmp = [x.strip().split(' ') for x in open(self.list_file)]
        if len(tmp[0]) == 3: # skip remove_missin for decording "raw_video label" type dataset_config
            if not self.test_mode:
                tmp = [item for item in tmp if int(item[1]) >= 8]
        self.video_list = [VideoRecord(item) for item in tmp]

        print('video number:%d' % (len(self.video_list)))

    def _sample_indices(self, start_idx, num_frames):
        if self.dense_sample:
            sample_pos = max(1, 1 + num_frames - self.sample_range)
            interval = self.sample_range // self.num_segments
            start_idx = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
            base_offsets = np.arange(self.num_segments) * interval
            offsets = (base_offsets + start_idx) % num_frames
            return np.array(offsets) + start_idx
        else:
            if num_frames <= self.total_length:
                if self.loop:
                    return np.mod(np.arange(
                        self.total_length) + randint(num_frames // 2),
                        num_frames) + start_idx
                offsets = np.concatenate((
                    np.arange(num_frames),
                    randint(num_frames,
                            size=self.total_length - num_frames)))
                return np.sort(offsets) + start_idx
            offsets = list()
            ticks = [i * num_frames // self.num_segments
                    for i in range(self.num_segments + 1)]

            for i in range(self.num_segments):
                tick_len = ticks[i + 1] - ticks[i]
                tick = ticks[i]
                if tick_len >= self.seg_length:
                    tick += randint(tick_len - self.seg_length + 1)
                offsets.extend([j for j in range(tick, tick + self.seg_length)])

            return np.array(offsets) + start_idx

    def _get_val_indices(self, start_idx, num_frames):
        if self.dense_sample:
            sample_pos = max(1, 1 + num_frames - self.sample_range)
            t_stride = self.sample_range // self.num_segments
            start_idx = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
            offsets = [(idx * t_stride + start_idx) % num_frames for idx in range(self.num_segments)]
            return np.array(offsets) + start_idx
        else:
            # tick = num_frames / float(self.num_segments)
            # offsets = [int(tick * x) % num_frames for x in range(self.num_segments)]

            ########## ATM implementation ###############
            seg_size = float(num_frames - 1) / self.num_segments
            offsets = []
            for i in range(self.num_segments):
                start = int(np.round(seg_size * i))
                frame_index = start + int(seg_size / 2)
                offsets.append(frame_index)	

            return np.array(offsets) + start_idx


    def _get_test_indices(self, start_idx, num_frames):
        if self.dense_sample:
            # multi-clip for dense sampling
            num_clips = self.test_clips
            sample_pos = max(0, num_frames - self.sample_range)
            interval = self.sample_range // self.num_segments
            start_list = [clip_idx * math.floor(sample_pos / (num_clips -1)) for clip_idx in range(num_clips)]
            base_offsets = np.arange(self.num_segments) * interval
            offsets = []
            for start_idx in start_list:
                offsets.extend((base_offsets + start_idx) % num_frames)
            return np.array(offsets) + start_idx
        else:
            # multi-clip for uniform sampling
            num_clips = self.test_clips
            seg_size = float(num_frames - 1) / self.num_segments
            seq = []
            duration = seg_size / (num_clips + 1)
            for temporal_sample_index in range(num_clips):
                for i in range(self.num_segments):
                    start = int(np.round(seg_size * i))
                    frame_index = start + int(duration * (temporal_sample_index + 1))
                    seq.append(frame_index)	
            return np.array(seq) + start_idx


    def _decord_decode(self, video_path):
        try:
            container = decord.VideoReader(video_path)
        except Exception as e:
            print("Failed to decode {} with exception: {}".format(
                video_path, e))
            return None
        
        return container

    def __getitem__(self, index):
        # decode frames to video_list
        if self.modality == 'video':
            _num_retries = 10
            for i_try in range(_num_retries):
                record = copy.deepcopy(self.video_list[index])
                directory = os.path.join(self.root_path, record.path)
                video_list = self._decord_decode(directory)
                # video_list = self._decord_pyav(directory)
                if video_list is None:
                    print("Failed to decode video idx {} from {}; trial {}".format(
                        index, directory, i_try)
                    )
                    index = random.randint(0, len(self.video_list))
                    continue
                break
        else:
            record = self.video_list[index]
            video_list = os.listdir(os.path.join(self.root_path, record.path))

        if not self.test_mode: # train/val
            segment_indices = self._sample_indices(record.start_idx, record.num_frames) if self.random_shift else self._get_val_indices(record.start_idx, record.num_frames) 
        else: # test
            segment_indices = self._get_test_indices(record.start_idx, record.num_frames)

        return self.get(record, video_list, segment_indices)


    def _load_image(self, directory, idx):
        if self.modality == 'RGB':
            try:
                return [Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(idx))).convert('RGB')]
            except Exception:
                print('error loading image:', os.path.join(self.root_path, directory, self.image_tmpl.format(idx)))
                return [Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(1))).convert('RGB')]


    def get(self, record, video_list, indices):
        images = list()
        for seg_ind in indices:
            p = int(seg_ind)
            if self.modality == 'video':
                seg_imgs = [Image.fromarray(video_list[p - 1].asnumpy()).convert('RGB')]
            else:
                seg_imgs = self._load_image(record.path, p)
            images.extend(seg_imgs)
            if p < len(video_list):
                p += 1

        record_label = {'verb': record.verb_label, 'noun': record.noun_label}
        if self.num_sample > 1:
            frame_list = []
            label_list = []
            for _ in range(self.num_sample):
                process_data, record_label = self.transform((images, record_label))
                frame_list.append(process_data)
                label_list.append(record_label)
            return frame_list, label_list
        else:
            process_data, record_label = self.transform((images, record_label))
            return process_data, record_label

    def __len__(self):
        return len(self.video_list)
