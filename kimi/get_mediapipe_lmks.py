import os
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

import mediapipe as mp
from pathlib import Path
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import FaceLandmarker,FaceLandmarkerOptions
import numpy as np
from pathlib import Path
from glob import glob
import cv2

videoPath = "/fslgroup/grp_lip/datasets/lrs2/mvlrs_v1/main/5551009007333662603/00001.mp4"

source_main_path = "/fslgroup/grp_lip/datasets/lrs2/mvlrs_v1/main/"
out_main_path = "/home/ksw38/groups/grp_landmarks/nobackup/archive/landmarks_mp/main"
folders = [f for f in os.listdir(source_main_path) if os.path.isdir(os.path.join(source_main_path, f))]

## Set Up ##
VisionRunningMode = mp.tasks.vision.RunningMode

__here__ = Path(__file__).parent
mdl_path = Path(f'{__here__}/..//data/face_landmarker_v2_with_blendshapes.task')
if not mdl_path.exists():
    mdl_path = Path(str(__here__) + '/data/face_landmarker_v2_with_blendshapes.task')
try:
    print("Using GPU delgate")
    from mp.tasks.BaseOptions import Delegate
    PYTHON_BASE_OPTIONS = python.BaseOptions(
        model_asset_path=mdl_path, delegate=mp.tasks.BaseOptions.Delegate.GPU)
    VisionRunningMode = mp.tasks.vision.RunningMode
except:
    print("GPU delegate not available, using CPU delegate.")
    PYTHON_BASE_OPTIONS = python.BaseOptions(
        model_asset_path=mdl_path)
    VisionRunningMode = mp.tasks.vision.RunningMode

def clearMediapipeInfo(includeBlendshapes=False, includeTransformation_FaceMesh=False):
    global DETECTOR, BASE_OPTIONS
    # SAVES COMPUTATION TIME INSIDE MEDIAPIPE BY NOT COMPUTING THINGS WE CON'T NEED.
    BASE_OPTIONS = FaceLandmarkerOptions(base_options=PYTHON_BASE_OPTIONS,
                                                output_face_blendshapes=includeBlendshapes,
                                                output_facial_transformation_matrixes=includeTransformation_FaceMesh,
                                                running_mode=VisionRunningMode.IMAGE,
                                                num_faces=1)
    DETECTOR = FaceLandmarker.create_from_options(BASE_OPTIONS)

clearMediapipeInfo(includeBlendshapes=False, includeTransformation_FaceMesh=False)

## Get Lmks ##

def get_lmks(image, convert2RGB=True):
    if convert2RGB:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
    results = DETECTOR.detect(image)
    if results.face_landmarks:
        lmks = results.face_landmarks[0] # 0 assumes single face
    else:
        lmks = None
    return lmks

def draw_landmarks(lmks_np, frame, save_path):
    for lm in lmks_np:
        h, w, _ = frame.shape
        x, y, z = lm
        x, y = int(x * w), int(y * h)
        cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
    cv2.imwrite(str(save_path), frame)

def lmks_np_for_video(videoPath):
    cap = cv2.VideoCapture(str(videoPath))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    video_lmks = []
    for frame in frames:
        frame = frame.copy()
        lmks = get_lmks(frame)
        if lmks is None:
            return None, frames
        lmks_np = np.array([[lm.x, lm.y, lm.z] for lm in lmks])
        video_lmks.append(lmks_np)
    video_lmks_np = np.stack(video_lmks)
    return video_lmks_np, frames


if __name__ == "__main__":
    num_skipped_files = 0
    skipped_files = []
    for folder in folders:
        source_folder_path = os.path.join(source_main_path, folder)
        files = glob(os.path.join(source_folder_path, "*.mp4"))
        for file in files:
            print(f"\n\n{file}\n\n")
            # Make File Paths
            source_path = os.path.join(source_main_path, folder, file)
            folder_path = os.path.join(out_main_path, folder)
            os.makedirs(folder_path, exist_ok=True)

            file_name = os.path.splitext(os.path.basename(file))[0]
            clearMediapipeInfo()
            video_lmks_np, frames = lmks_np_for_video(file)
            if video_lmks_np is None:
                num_skipped_files += 1
                skipped_files.append(file)
                continue


            out_path_lmks = os.path.join(out_main_path, folder, file_name)
            np.save(out_path_lmks, video_lmks_np)

    print(f"\n{num_skipped_files} were skipped")
    print(skipped_files)  

    save_path = "mediapipe_lmks.jpg"
    idx = 0
    draw_landmarks(video_lmks_np[idx], frames[idx], save_path)
    print(f"Example image saved at {save_path}")