import numpy as np

load_path = "/home/ksw38/groups/grp_landmarks/nobackup/archive/landmarks/main/5535415699068794046/00001_lmks.npy"
load_path = "/home/ksw38/groups/grp_landmarks/nobackup/archive/landmarks/main/5535415699068794046/00001_roll.npy"
load_path = "/home/ksw38/groups/grp_landmarks/nobackup/archive/landmarks/main/5535423430009926848/00001_pitch.npy"
load_path = "/home/ksw38/groups/grp_landmarks/nobackup/archive/landmarks/main/5535423430009926848/00003_yaw.npy"
load_path = "/home/ksw38/groups/grp_landmarks/nobackup/archive/landmarks/main/6149543623795728976/00035_yaw.npy"
data = np.load(load_path)

print(data)

import os
main_path = "/home/ksw38/groups/grp_landmarks/nobackup/archive/landmarks/main"
folders = [f for f in os.listdir(main_path) if os.path.isdir(os.path.join(main_path, f))]
# print(folders[len(folders) - 1])
# print(len(folders))
folders_num = [int(f) for f in folders]
# print(folders_num[-1])
# print(max(folders_num))

source_main_path = "/fslgroup/grp_lip/datasets/lrs2/mvlrs_v1/main/"
source_folders = [f for f in os.listdir(source_main_path) if os.path.isdir(os.path.join(source_main_path, f))]
source_folders_num = [int(f) for f in source_folders]

folders_num = sorted(folders_num)

for i, folder in enumerate(folders_num):
    if folder != source_folders_num[i]:
        print(i)
        print(folder)
        print(source_folders[i])

print(len(folders))
print(folders_num[-1])
print(source_folders[2424:len(source_folders)])
print(source_folders[-1])