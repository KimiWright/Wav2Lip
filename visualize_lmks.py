from os.path import dirname, join, basename, isfile
from torch.utils import data as data_utils
from glob import glob
import numpy as np
import torch
import re
import os
import cv2

from hparams import hparams, get_image_list

video_root = '/home/ksw38/groups/grp_lip/nobackup/autodelete/datasets/fslgroup/grp_lip/compute/datasets/LRS2/preprocessedRetinaface/lrs2/lrs2_video_seg24s/mvlrs_v1/main/'
# data_root = '/home/ksw38/groups/grp_landmarks/nobackup/archive/landmarks_preprocessed/main'
data_root = '/home/ksw38/groups/grp_landmarks/nobackup/archive/landmarks/main'
visual_root = '/home/ksw38/RVL/color_syncnet/Wav2Lip/lrs2_preprocessed'
visual_root = "/fslgroup/grp_lip/datasets/lrs2/mvlrs_v1/main/"

syncnet_T = 5

class Dataset(object):
    def __init__(self, split):
        # self.all_videos = get_npy_list(args.data_root, split)
        self.all_videos = get_image_list(video_root, split)
        self.idx = -1

    def get_frame_id(self, frame):
        # return int(basename(frame).split('.')[0][0:ID_LEN])
        frame_name = basename(frame).split('.')[0]
        frame_digits = re.sub(r'\D', '', frame_name)
        return int(frame_digits)


    def __len__(self):
        return len(self.all_videos)

    def __getitem__(self, idx):
        # Syncnet is set up randomly sync or not sync a video, that is part of why they take out 5 frame chunks
        while 1:
            self.idx += 1
            idx = self.idx

            # find the path to the video at index idx
            vidname = self.all_videos[idx]
            # keep the path and filename of the video, but remove the extension (for finding the .wav file)
            vidname_no_ext = os.path.splitext(vidname)[0]

            # 5 digit id
            vidname_file = os.path.splitext(os.path.basename(vidname))[0]
            # video and landmarks folder name (log numberical id)
            vidname_folder = os.path.basename(os.path.dirname(vidname))
            # landmarks file with the 5 digit id, but not the lmks, roll, pitch, yaw endings
            npy_head = join(data_root, vidname_folder, vidname_file)
            visual_head = join(visual_root, vidname_folder, vidname_file)


            img_list = glob(os.path.join(visual_head, '*.jpg'))
            if len(img_list) == 0:
                videoPath = visual_head + ".mp4"
                if os.path.exists(videoPath):
                    cap = cv2.VideoCapture(str(videoPath))
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        img_list.append(frame)
                    cap.release()
                else:
                    continue


            # get all of the npy files corresponding to the video
            npy_files = []
            endings = ['_lmks.npy', '_roll.npy', '_pitch.npy', '_yaw.npy']
            for ending in endings:
                npy_file = npy_head + ending
                if not isfile(npy_file):
                    continue
                npy_files.append(npy_file)

            # print(f"img_list {len(img_list)}")
            # print(f"npy_files {len(npy_files)}")
            # print(visual_head)
            # print(npy_head)

            # retrive the data from the npy files
            npy_data = []
            for npy_file in npy_files:
                try:
                    npy_data.append(np.load(npy_file))
                except Exception as e:
                    print(f"Error loading npy file {npy_file}: {e}")
                    break
            if len(npy_data) != 4:
                continue

            num_frames = npy_data[0].shape[0]

            if num_frames <= 3 * syncnet_T:
                continue

            window_fnames = []
            for npy_datum in npy_data:
                # get the window of npy data from start_id to start_id + syncnet_T
                window_npy = npy_datum
                if window_npy is None:
                    break
                window_fnames.append(window_npy)
            if len(window_fnames) != 4:
                continue

            # print(visual_head)
            # print(npy_files)
            # print(img_list)
            return window_fnames, img_list
        
def visualize(frame, lmks, name="visualized_lmks.jpg"):
    height, width, channels = frame.shape
    # x_vals = (lmks[0] + 1 ) / 2 * width
    # y_vals = (lmks[1] + 1 ) / 2 * height
    # x_vals = lmks[0]
    # y_vals = lmks[1]
    scale = 50
    x_vals = (lmks[0] * scale + 1 ) / 2 * width
    y_vals = (lmks[1] * scale + 1 ) / 2 * height

    print(x_vals)
    print('~~~~~~~~~~~~~~~~~~~~')
    print(y_vals)
    print()
    print(width, height)

    if len(x_vals) != len(y_vals):
        raise(ValueError("x and y are different lengths"))
    
    for i in range(len(x_vals)):
        x = x_vals[i]
        y = y_vals[i]
        cv2.circle(frame, (int(x), int(y)), 1, (0, 255, 0), -1)
    
    cv2.imwrite(name, frame)
    print(f"Saved frame at {name}")
        
if __name__ == "__main__":
    data_limit = 4
    data_start = None
    batch_size = 1 # hparams.syncnet_batch_size
    test_dataset = Dataset('val')

    test_data_loader = data_utils.DataLoader(
        test_dataset, batch_size=batch_size,
        num_workers=1)
    
    # prog_bar = tqdm(enumerate(test_data_loader))
    prog_bar = enumerate(test_data_loader)
    for step, (x_list, img_list) in prog_bar:
        if data_start is not None and step < data_start:
            continue
        print(f"Step: {step}")
        x_lmks = x_list[0]
        x_roll = x_list[1]
        x_pitch = x_list[2]
        x_yaw = x_list[3]

        print(x_lmks.shape)
        print(x_roll.shape, x_pitch.shape, x_yaw.shape)
        print(f"img_list {len(img_list)}")
        print()

        if data_limit is not None and step > data_limit:
            break
    
    # print("***Visualizing Landmarks***")
    idx = 0
    batch_idx = 0
    frame = np.array(img_list[idx][batch_idx])
    visualize(frame, x_lmks[batch_idx][idx])
    print("End")