'''
import joblib

data = joblib.load("/hdd/xinpeng/PHC/data/amass/pkls/amass_isaac_im_train_take6_upright_slim.pkl")

if isinstance(data, dict):
    print("Keys in the loaded file:")
    for key in data.keys():
        print(key)
else:
    print("The loaded data is not a dictionary.")
  '''


import joblib
import matplotlib.pyplot as plt

pkl_file_path = '/hdd/xinpeng/PHC/data/amass/pkls/amass_isaac_im_train_take6_upright_slim.pkl'

data = joblib.load(pkl_file_path)

selected_datasets = [
    entry for entry in data.keys()  
    if any(keyword in entry.lower() for keyword in ['walk'])
]

dataset_with_frames = []

for dataset_name in selected_datasets:
    dataset = data[dataset_name]
    if 'pose_quat_global' in dataset:
        pose_quat_global = dataset['pose_quat_global']
        dataset_with_frames.append((dataset_name, pose_quat_global.shape[0]))

# 按照帧数（pose_quat_global.shape[0]）进行降序排序
dataset_with_frames.sort(key=lambda x: x[1], reverse=True)
for i, (dataset_name, frame_count) in enumerate(dataset_with_frames[100:200], start=101):
    print(f"Dataset {i}: {dataset_name} - Frame count: {frame_count}")


'''
import joblib

pkl_file_path = '/hdd/xinpeng/PHC/data/amass/pkls/amass_isaac_im_train_take6_upright_slim.pkl'

data = joblib.load(pkl_file_path)

dataset_names = list(data.keys())

selected_datasets = [
    name for name in dataset_names if any(keyword in name for keyword in ["walk"])
]

dataset_name = selected_datasets[0]  
dataset = data[dataset_name]
pose_quat_global = dataset.get('pose_quat_global', None)

if pose_quat_global is None:
    print(f"No 'pose_quat_global' data found in dataset {dataset_name}")
else:
    print(f"Shape of 'pose_quat_global': {pose_quat_global.shape}")

'''