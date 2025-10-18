#!/usr/bin/env python3
from pathlib import Path
import argparse
import numpy as np
import cv2
import sys
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = ""
print(str(Path(__file__).resolve().parent))
from facetools import genMediapipeInfo, norm_lmks, clearMediapipeInfo
#print(facetools.__file__)
from tqdm import tqdm

def norm(lmks):
    lmks_norm = np.zeros_like(lmks, dtype=np.float32)
    for t in range(lmks.shape[0]):
        frame = lmks[t]
        min_xy = frame.min(axis=0)
        max_xy = frame.max(axis=0)
        scale = (max_xy - min_xy).max() / 2.0
        center = 0  # (max_xy + min_xy) / 2.0
        if scale < 1e-6:
            scale = 1.0
        lmks_norm[t] = (frame - center) / scale
    return lmks_norm

def parse_args():
    p = argparse.ArgumentParser(
        description="Process MP4s into landmarks, split by array index for parallel runs."
    )
    p.add_argument("--index", type=int, required=True, help="This worker's index (e.g., SLURM_ARRAY_TASK_ID).")
    p.add_argument("--groups", type=int, required=True, help="Total number of workers in the group.")
    p.add_argument(
        "--src",
        type=Path,
        default=Path("/fslgroup/grp_lip/datasets/lrs2/mvlrs_v1/pretrain"),
        help="Source root directory."
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("/home/ksw38/groups/grp_landmarks/nobackup/archive/landmarks_norm/pretrain"),
        help="Output root directory."
    )
    p.add_argument(
        "--exclude-train",
        action="store_true",
        help='Exclude files with "train" in the filename (same as original behavior).'
    )
    return p.parse_args()

def main():
    args = parse_args()

    if args.index < 0 or args.groups <= 0 or args.index >= args.groups:
        print(f"Invalid index/groups combination: index={args.index}, groups={args.groups}", file=sys.stderr)
        sys.exit(2)

    # Stable, deterministic list of files
    files = sorted(args.src.rglob("*.mp4"))

    # Match original behavior: remove files with "train" in the name (enabled by default for parity)
    if args.exclude_train or True:
        files = [f for f in files if "train" not in f.name]

    # Even, interleaved split across workers (good balance even if classes are clustered)
    my_files = [f for i, f in enumerate(files) if i % args.groups == args.index]

    num_skipped_files = 0
    skipped_files = []

    for file in tqdm(my_files, desc=f"worker {args.index}/{args.groups}"):
        try:
            rel_path = file.relative_to(args.src)
            out_folder = args.out / rel_path.parent
            out_folder.mkdir(parents=True, exist_ok=True)

            file_name = file.stem
            out_path_lmks = out_folder / f"{file_name}_lmks.npy"
            out_path_yaw = out_folder / f"{file_name}_yaw.npy"
            out_path_pitch = out_folder / f"{file_name}_pitch.npy"
            out_path_roll = out_folder / f"{file_name}_roll.npy"

            # Load video
            cap = cv2.VideoCapture(str(file))
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            cap.release()

            # Extract landmarks and angles
            _, lmks, allYaw, allPitch, allRoll = genMediapipeInfo(frames)
            clearMediapipeInfo()

            try:
                lmks = np.swapaxes(lmks, 1, 2)
            except Exception as e:
                print(f"\nError: {e}\non file {file}\n")
                num_skipped_files += 1
                skipped_files.append(str(file))
                continue

            lmks = norm(lmks)

            # Save outputs
            np.save(out_path_lmks, lmks)
            np.save(out_path_yaw, allYaw)
            np.save(out_path_pitch, allPitch)
            np.save(out_path_roll, allRoll)

        except Exception as e:
            print(f"\nUnexpected error processing {file}: {e}")
            num_skipped_files += 1
            skipped_files.append(str(file))
            continue

    print(f"\nWorker {args.index}: processed {len(my_files)} files, skipped {num_skipped_files}")
    if skipped_files:
        print("Skipped files:")
        for s in skipped_files:
            print(s)

if __name__ == "__main__":
    main()
