import os
import h5py

# path = r"C:\Users\Kimi\.cache\kagglehub\datasets\adrianlubitz\vvadlrs3\versions\4"
path = r"/home/ksw38/.cache/kagglehub/datasets/adrianlubitz/vvadlrs3/versions/4"
files = os.listdir(path)

face_img_h5_path = os.path.join(path, files[1])
print(face_img_h5_path)

with h5py.File(face_img_h5_path, 'r') as f:
    # List all groups
    print("Keys: %s" % f.keys())
    # # Get the first group
    # a_group_key = list(f.keys())[0]
    # # Get the data
    # data = list(f[a_group_key])
    # print(data)
    # # Get the data shape
    # data_shape = f[a_group_key].shape
    # print(data_shape)
    x_test = f['x_test']
    x_train = f['x_train']
    y_test = f['y_test']
    y_train = f['y_train']
    print(x_test.shape)
    print(x_train.shape)
    print(y_test.shape)
    print(y_train.shape)


