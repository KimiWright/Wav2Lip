from hparams import hparams, get_image_list
import os
import re
from collections import defaultdict

import pandas as pd

print("CSV numbers")
def num_rows_csv(csv_file):
    df = pd.read_csv(csv_file)
    num_rows = len(df)
    print(num_rows)

# My csv (too short)
# num_rows_csv("/home/ksw38/groups/grp_landmarks/nobackup/autodelete/LRS2/preprocessedRetinaface/labels/lmks_lrs2_train-val_transcript_lengths_seg24s.csv")
# Shad's csv (correct)
num_rows_csv("/home/ksw38/groups/grp_lip/nobackup/archive/datasets/LRS2/preprocessedRetinaface/labels/lrs2_train_transcript_lengths_seg24s.csv")

num_rows_csv("/home/ksw38/groups/grp_lip/nobackup/archive/datasets/LRS2/preprocessedRetinaface/labels/lrs2_train_transcript_lengths_seg24s.csv")
num_rows_csv("/home/ksw38/groups/grp_lip/nobackup/archive/datasets/LRS2/preprocessedRetinaface/labels/lrs2_train_transcript_lengths_seg24s_0to100.csv")

# Unfiltered
print("unfiltered")
num_rows_csv("/home/ksw38/groups/grp_landmarks/nobackup/autodelete/labels/lrs2_train_transcript_lengths_seg24s.csv")
num_rows_csv("/home/ksw38/groups/grp_lip/nobackup/archive/datasets/LRS2/preprocessedRetinaface/labels/lrs2_test_transcript_lengths_seg24s.csv")
print("filtered")
# num_rows_csv("/home/ksw38/groups/grp_landmarks/nobackup/autodelete/LRS2/preprocessedRetinaface/labels/lrs2_train_transcript_lengths_seg24s.csv")
# num_rows_csv("/home/ksw38/groups/grp_landmarks/nobackup/autodelete/LRS2/preprocessedRetinaface/labels/lrs2_test_transcript_lengths_seg24s.csv")

# num_rows_csv("/home/ksw38/groups/grp_landmarks/nobackup/autodelete/labels/trial_lrs2_train_transcript_lengths_seg24.csv")
num_rows_csv("/home/ksw38/groups/grp_landmarks/nobackup/autodelete/LRS2/preprocessedRetinaface/labels/lmks_lrs2_train_transcript_lengths_seg24.csv")
num_rows_csv("/home/ksw38/groups/grp_landmarks/nobackup/autodelete/LRS2/preprocessedRetinaface/labels/lrs2_train_transcript_lengths_seg24.csv")


# lmks_path = "/home/ksw38/groups/grp_landmarks/nobackup/autodelete/LRS2/preprocessedRetinaface/labels/lmk_train-val_transcript_lengths_seg24s.csv"
# short_path = "/home/ksw38/groups/grp_landmarks/nobackup/autodelete/LRS2/preprocessedRetinaface/labels/lrs2_train-val_transcript_lengths_seg24s_0to100.csv"

# num_rows_csv(lmks_path)
# num_rows_csv(short_path)
ORIGINAL_PATH_TEST =  "/home/ksw38/groups/grp_lip/nobackup/archive/datasets/LRS2/preprocessedRetinaface/labels/lrs2_test_transcript_lengths_seg24s.csv" #"/home/ksw38/groups/grp_lip/nobackup/archive/datasets/LRS2/preprocessedRetinaface/labels/lrs2_test_transcript_lengths_seg24s.2.0.csv"
ORIGINAL_PATH_TRAIN_VAL = "/home/ksw38/groups/grp_lip/nobackup/archive/datasets/LRS2/preprocessedRetinaface/labels/lrs2_train_transcript_lengths_seg24s.csv"
num_rows_csv(ORIGINAL_PATH_TEST)
num_rows_csv(ORIGINAL_PATH_TRAIN_VAL)

folderA = '/home/ksw38/groups/grp_landmarks/nobackup/archive/landmarks_mp/main/'
folderB = "/home/ksw38/groups/grp_landmarks/nobackup/archive/landmarks_norm/main"

def count_videos(root_dir, extensions=(".mp4", ".avi", ".mov", ".mkv", ".npy")):
    count = 0
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(extensions):
                count += 1
    return count

SUFFIXES = ("lmks", "yaw", "roll", "pitch")

def rel_from_main(root, base_root):
    """Get path relative to the 'main' subdir (skip differing top parts)."""
    rel = os.path.relpath(root, base_root)
    parts = rel.split(os.sep)
    if "main" in parts:
        idx = parts.index("main")
        return os.path.join(*parts[idx:])  # everything after 'main'
    return rel

def collect_A(folderA):
    base_counts = defaultdict(set)
    pat = re.compile(r"^(?P<base>.+?)_(?P<suffix>lmks|yaw|roll|pitch)\.npy$", re.IGNORECASE)

    for root, _, files in os.walk(folderA):
        rel_dir = rel_from_main(root, folderA)
        for f in files:
            m = pat.match(f)
            if m:
                base, suffix = m.groups()
                base_counts[(rel_dir, base)].add(suffix)

    complete = {k for k, v in base_counts.items() if all(s in v for s in SUFFIXES)}
    incomplete = {k: tuple(s for s in SUFFIXES if s not in v) for k, v in base_counts.items() if k not in complete}
    return complete, incomplete

def collect_B(folderB):
    bases = set()
    skip_pat = re.compile(r"_(lmks|yaw|roll|pitch)\.npy$", re.IGNORECASE)
    pat = re.compile(r"^(?P<base>.+?)\.npy$", re.IGNORECASE)

    for root, _, files in os.walk(folderB):
        rel_dir = rel_from_main(root, folderB)
        for f in files:
            if not f.endswith(".npy") or skip_pat.search(f):
                continue
            m = pat.match(f)
            if m:
                bases.add((rel_dir, m.group("base")))
    return bases

def compare(folderA, folderB):
    setA, incomplete = collect_A(folderA)
    setB = collect_B(folderB)

    onlyA = sorted(setA - setB)
    onlyB = sorted(setB - setA)

    groupA = defaultdict(list)
    for rel_dir, base in onlyA:
        groupA[rel_dir].append(base)

    groupB = defaultdict(list)
    for rel_dir, base in onlyB:
        groupB[rel_dir].append(base)

    for d in (groupA, groupB):
        for k in d:
            d[k].sort()

    return groupA, groupB, incomplete


# Example usage
folderA = "/home/ksw38/groups/grp_landmarks/nobackup/archive/landmarks_norm"
folderB = "/home/ksw38/groups/grp_landmarks/nobackup/archive/landmarks_mp"

onlyA, onlyB, incomplete = compare(folderA, folderB)

# print("=== Present in A but missing in B ===")
# for d, bases in sorted(onlyA.items()):
#     print(f"[{d}] ({len(bases)})")
#     for b in bases[:10]:  # preview first 10
#         print("  ", b)
#     if len(bases) > 10:
#         print("  ...")

# print("\n=== Present in B but missing in A ===")
# for d, bases in sorted(onlyB.items()):
#     print(f"[{d}] ({len(bases)})")
#     for b in bases[:10]:
#         print("  ", b)
#     if len(bases) > 10:
#         print("  ...")

# print("\nIncomplete sets in A:", len(incomplete))

# total_missing = sum(len(b) for b in onlyA.values())
# print("Total missing in A:", total_missing)

# for i, (rel_dir, bases) in enumerate(onlyA.items()):
#     if i >= 5:
#         break
#     print(f"[{rel_dir}] ({len(bases)} missing)")
#     print("  Examples:", bases[:5])

print("Get Image List")
video_root = '/home/ksw38/groups/grp_lip/nobackup/autodelete/datasets/fslgroup/grp_lip/compute/datasets/LRS2/preprocessedRetinaface/lrs2/lrs2_video_seg24s/mvlrs_v1/main/'

all_videos = get_image_list(video_root, 'val')
print(len(all_videos))
all_videos = get_image_list(video_root, 'train')
print(len(all_videos))

print("Mp")
data_root = '/home/ksw38/groups/grp_landmarks/nobackup/archive/landmarks_mp/main/'
count = count_videos(data_root)
print(count)

# data_root = '/home/ksw38/groups/grp_landmarks/nobackup/archive/landmarks_preprocessed/main'
# count = count_videos(data_root)
# print(count)

# data_root = '/home/ksw38/groups/grp_landmarks/nobackup/archive/landmarks/main'
# count = count_videos(data_root)
# print(count)

print("norm")
data_root = "/home/ksw38/groups/grp_landmarks/nobackup/archive/landmarks_norm/main"
count = count_videos(data_root)
print(count)

data_root = "/home/ksw38/groups/grp_landmarks/nobackup/archive/landmarks_norm/pretrain"
count = count_videos(data_root)
print(count)

data_root = "/home/ksw38/groups/grp_landmarks/nobackup/archive/landmarks_norm/pretrain_partial"
count = count_videos(data_root)
print(count)

# data_root = '/home/ksw38/groups/grp_lip/nobackup/autodelete/datasets/fslgroup/grp_lip/compute/datasets/LRS2/preprocessedRetinaface/lrs2/lrs2_video_seg24s/mvlrs_v1/main/'
# count = count_videos(data_root)
# print(count)

print("/fslgroup/grp_lip/datasets/lrs2/mvlrs_v1/main/")
data_root = "/fslgroup/grp_lip/datasets/lrs2/mvlrs_v1/main/"
count = count_videos(data_root)
print(count)

data_root = "groups/grp_lip/nobackup/autodelete/datasets/fslgroup/grp_lip/compute/datasets/LRS2/preprocessedRetinaface/lrs2/lr2_video_seg24s/mlrs_v1/main"
data_root = "/home/ksw38/groups/grp_lip/nobackup/autodelete/datasets/fslgroup/grp_lip/compute/datasets/LRS2/preprocessedRetinaface/lrs2/lrs2_video_seg24s/mvlrs_v1/main"
print(data_root)
count = count_videos(data_root)
print(count)

data_root = "/fslgroup/grp_lip/datasets/lrs2/mvlrs_v1/pretrain"
print(data_root)
count = count_videos(data_root)
print(count)

data_root = "/home/ksw38/groups/grp_landmarks/nobackup/archive/landmarks_norm/pretrain"
print(data_root)
count = count_videos(data_root)
print(count)

data_root = "/home/ksw38/groups/grp_landmarks/nobackup/archive/landmarks_norm/pretrain_partial"
data_root = "/home/ksw38/groups/grp_landmarks/nobackup/autodelete/pretrain_partial"
print(data_root)
count = count_videos(data_root)
print(count)

data_root = "/home/ksw38/groups/grp_landmarks/nobackup/archive/landmarks_norm/pretrain_2"
print(data_root)
count = count_videos(data_root)
print(count)

# import torch

# checkpoint_path_wav2lip = "/home/ksw38/RVL/color_syncnet/Wav2Lip/lipsync_expert.pth"
# checkpoint_path_mine = "/home/ksw38/RVL/color_syncnet/Wav2Lip/checkpoints/checkpoint_step000510000.pth"

# checkpoint_wav2lip = torch.load(checkpoint_path_wav2lip, map_location='cpu')
# state_dict_wav2lip = checkpoint_wav2lip['state_dict']

# checkpoint_mine = torch.load(checkpoint_path_mine, map_location='cpu')
# state_dict_mine = checkpoint_mine['state_dict']

# # for k in full_state_dict.keys():
# #     print(k)

# keys_mine = list(state_dict_mine.keys())
# keys_wav2lip = list(state_dict_wav2lip.keys())

# mismatched = 0
# for i in range(len(keys_mine)):
#     print(f"{keys_mine[i]}   |   {keys_wav2lip[i]}")
#     if keys_mine[i] != keys_wav2lip[i]:
#         print("\tMISMATCH")
#         mismatched += 1
#     # keys_mine_val = keys_mine[i].split('.')
#     # keys_wav2lip_val = keys_wav2lip[i].split('.')
#     # for j in range(1,5):
#     #     if keys_mine_val[-j] != keys_wav2lip_val[-j]:
#     #         print(f'\tMISMATCH at -{j}')
#     #         mismatched += 1
# print(f"\n{mismatched} keys were mismatched")
# print(len(keys_mine))
# print(len(keys_wav2lip))