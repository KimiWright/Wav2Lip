import h5py

source_main_path = "/home/ksw38/.cache/kagglehub/datasets/adrianlubitz/vvadlrs3/versions/4/faceImages_small.h5"
start_frame_num = 0
syncnet_T = 5

with h5py.File(source_main_path, 'r') as f:
    # Get frames from the h5 file
    x_test = f['x_test']

    for i, frames in enumerate(x_test):
        frames = frames[start_frame_num:start_frame_num+syncnet_T]
        print(frames.shape)  # Print the shape of the frames