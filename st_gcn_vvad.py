import numpy as np
from tqdm import tqdm
import re
import os
import torch
from torch import optim
import torch.utils.data as data_utils
import torch.nn.functional as F



data_root = "/home/ksw38/groups/grp_landmarks/nobackup/autodelete/landmarks_vvadlrs3_norm/main"
x_test_path = os.path.join(data_root, "x_test")
y_test_path = os.path.join(data_root, "y_test.npy")
x_train_path = os.path.join(data_root, "x_train")
y_train_path = os.path.join(data_root, "y_train.npy")

syncnet_T = 5
data_limit = None

def get_files_list(folder_path):
    print(folder_path)
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    files = sorted(files, key=lambda x: int(re.search(r'(\d+)', x).group(0)))
    return files

def find_file_by_idx(files, idx):
    """Return the first file whose name contains the idx (as a full number)."""
    idx_str = str(idx)
    for f in files:
        base = os.path.basename(f)
        # Make sure we match the whole number, not just substring
        if f"_{idx_str}." in base or base.startswith(idx_str + "_"):
            return f
    return None

def get_window_npy(x, x_rot, syncnet_T = 5, start_id=0):
    if start_id + syncnet_T < len(x):
        x_rot = np.swapaxes(x_rot, 0, 1)
        x_rot = x_rot[start_id : start_id + syncnet_T]
        x_rot = np.swapaxes(x_rot, 0, 1)
        return x[start_id : start_id + syncnet_T], x_rot
    else:
        return None

def get_data(data_root, ground_truth, data_point_limit=None, start_idx=0):
    all_files = get_files_list(data_root) 
    lmks_files = [f for f in all_files if f.endswith('lmks.npy')]
    roll_files = [f for f in all_files if f.endswith('roll.npy')]
    pitch_files = [f for f in all_files if f.endswith('pitch.npy')]
    yaw_files = [f for f in all_files if f.endswith('yaw.npy')]   
    y = np.load(ground_truth)
    
    data = []
    for idx in tqdm(range(len(lmks_files))):
        if idx < start_idx:
            continue

        npy_file_names = find_file_by_idx(all_files, idx)
        lmks_file = find_file_by_idx(lmks_files, idx)
        roll_file = find_file_by_idx(roll_files, idx)
        pitch_file = find_file_by_idx(pitch_files, idx)
        yaw_file = find_file_by_idx(yaw_files, idx)

        npy_file_names = [lmks_file, roll_file, pitch_file, yaw_file]

        if any(file is None for file in npy_file_names):
            continue
        
        npy_files = [os.path.join(data_root, f) for f in npy_file_names]
        npy_data = []
        for npy_file in npy_files:
            try:
                npy_data.append(np.load(npy_file))
            except Exception as e:
                # print(f"Error loading npy file {npy_file}: {e}")
                break
        if len(npy_data) != 4:
            continue
        # Check if the data is empty
        if any(data.size == 0 for data in npy_data):
            # print(f"Empty data in npy files: {npy_file_names}")
            continue
        
        num_frames = npy_data[0].shape[0]
        if num_frames < syncnet_T:
            continue
            
        x_lmks = npy_data[0]
        x_roll = npy_data[1]
        x_pitch = npy_data[2]
        x_yaw = npy_data[3]

        min_frames = 5 # minimum number of frames for the kernel size
        x_rot = torch.FloatTensor(np.vstack((x_roll, x_pitch, x_yaw)))
            
        x_lmks = np.swapaxes(x_lmks, 1,2)
        x = torch.FloatTensor(x_lmks)

        data.append((x, x_rot, y[idx]))
        if data_point_limit is not None and len(data) >= data_point_limit:
            break

    return data

# data = get_data(x_test_path, y_test_path, data_limit)
data_out_test = []
data_out_train = []

# for datum in data:
#     x, x_rot, y = datum
#     x, x_rot = get_window_npy(x, x_rot)
#     print(x.shape, x_rot.shape, y)

class Dataset_Frames(object):
    def __init__(self, split = 'test', frames=syncnet_T, data = None, data_point_limit=data_limit):
        if data == None:
            if split == 'test':
                global data_out_test
                self.data = data_out_test
                if len(self.data) == 0:
                    data_out_test = get_data(x_test_path, y_test_path, data_point_limit=data_point_limit)
                    self.data = data_out_test
            elif split == 'train':
                global data_out_train
                self.data = data_out_train
                if len(self.data) == 0:
                    data_out_train = get_data(x_train_path, y_train_path, data_point_limit=data_point_limit)
                    self.data = data_out_train
            else:
                raise ValueError("Split must be 'test' or 'train'")
        else:
            self.data = data
        self.processed_data = []
        missing_window_num = 0
        for datum in self.data:
            x_full, x_rot_full, y = datum
            if x_full is None:
                raise ValueError("x_video_full is None")
            window = get_window_npy(x_full, x_rot_full, syncnet_T=frames, start_id=0)
            if window is None:
                missing_window_num += 1
                continue
            x, x_rot = window
            if x is not None:
                self.processed_data.append((x, x_rot, y))
        print(f"Missing {missing_window_num} windows")
        
    def __len__(self):
        return len(self.processed_data)
    def __getitem__(self, idx):
        return self.processed_data[idx]
        
if __name__ == "__main__":
    data_limit = 4
    batch_size = 1

    test_dataset=Dataset_Frames("test", 7)
    # train_dataset=Dataset_Frames("train")
    # wildcard_data_root = '/home/ksw38/groups/grp_landmarks/nobackup/archive/landmarks_vvadlrs3/main/x_test/'
    # wildcard_truth = '/home/ksw38/groups/grp_landmarks/nobackup/archive/landmarks_vvadlrs3/main/y_test.npy'
    # wildcard_data = get_data(wildcard_data_root, wildcard_truth, data_limit, 100)
    # wildcard_dataset=Dataset_Frames('test',syncnet_T,wildcard_data)

    test_data_loader = data_utils.DataLoader(
        test_dataset, batch_size=batch_size,
        num_workers=8)
    
    with torch.no_grad():
        prog_bar = enumerate(test_data_loader)
        for step, (x, x_rot, y) in prog_bar:
            print(x.shape, x_rot.shape, y)