import os

video_path = "/home/ksw38/groups/grp_lip/datasets/lrs2/mvlrs_v1/main/5535415699068794046/00001.mp4"
landmarks_path = "/home/ksw38/groups/grp_landmarks/nobackup/archive/landmarks/main/5535415699068794046/00001_lmks.npy"

def readable_size(num_bytes):
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if num_bytes < 1024:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024


checking_path = video_path
size_bytes = os.path.getsize(checking_path)
print(size_bytes)
print(readable_size(size_bytes))

checking_path = landmarks_path
size_bytes = os.path.getsize(checking_path)
print(size_bytes)
print(readable_size(size_bytes))