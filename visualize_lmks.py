import run_rotation_statistics as run_stats

data_root = '/home/ksw38/groups/grp_landmarks/nobackup/archive/landmarks_vvadlrs3/main/x_test/'
ground_truth = '/home/ksw38/groups/grp_landmarks/nobackup/archive/landmarks_vvadlrs3/main/y_test.npy'
data = run_stats.get_data(data_root=data_root, ground_truth=ground_truth, data_point_limit=1, start_idx=0)

datum, y = data[0]

(x_video, x_roll, x_pitch, x_yaw) = datum
# print(x_video)
# print(x_roll)
# print(x_pitch)
# print(x_yaw)

x_lmks = x_video[:, :-3]   # all but last 3 columns
x_roll_recovered  = x_video[:, -3]
x_pitch_recovered = x_video[:, -2]
x_yaw_recovered   = x_video[:, -1]

print(x_lmks.shape)
print(x_lmks)