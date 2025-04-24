from pathlib import Path
import cv2
import numpy as np
videoPath = Path("/fslgroup/grp_lip/datasets/lrs2/mvlrs_v1/main/5551009007333662603/00001.mp4")
cap = cv2.VideoCapture(str(videoPath))
frames = []
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(frame)
cap.release()
from facetools import genMediapipeInfo,norm_lmks
_, lmks, allYaw, allPitch, allRoll = genMediapipeInfo(frames) #this does the landmark extraction and a bunch of normalization
lmks = norm_lmks(lmks) # this does the final normalization
#save the lmks and the yaw, pitch, roll to a filefrom facetools import genMediapipeInfo,norm_lmks
print("\n\nlmks\n")
print(lmks)
print("\n\nyaw\n")
print(allYaw)
print("\n\npitch\n")
print(allPitch)
print("\n\nRoll\n")
print(allRoll)

