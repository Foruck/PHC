'''
import joblib
import matplotlib.pyplot as plt

pkl_file_path = '/hdd/xinpeng/PHC/data/amass/pkls/amass_isaac_im_train_take6_upright_slim.pkl'

data = joblib.load(pkl_file_path)

selected_datasets = [
    entry for entry in data.keys()  
    if any(keyword in entry.lower() for keyword in ['jog'])
]

dataset_with_frames = []

for dataset_name in selected_datasets:
    dataset = data[dataset_name]
    if 'pose_quat_global' in dataset:
        pose_quat_global = dataset['pose_quat_global']
        dataset_with_frames.append((dataset_name, pose_quat_global.shape[0]))

# 按照帧数（pose_quat_global.shape[0]）进行降序排序
dataset_with_frames.sort(key=lambda x: x[1], reverse=True)
for i, (dataset_name, frame_count) in enumerate(dataset_with_frames[:100], start=0):
    print(f"Dataset {i}: {dataset_name} - Frame count: {frame_count}")



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


import joblib
import os

pkl_file_path = '/hdd/xinpeng/PHC/data/amass/pkls/amass_isaac_im_train_take6_upright_slim.pkl'

output_pkl_path = '/hdd/xinpeng/PHC/phc/muchun_test/selected_motion/0-DFaust_50026_50026_running_on_spot_poses.pkl'

data = joblib.load(pkl_file_path)

selected_names = [
    #"0-SFU_0005_0005_Walking001_poses",
    #"0-SFU_0007_0007_Walking001_poses",
    #"0-HumanEva_S1_Walking_3_poses",
    #"0-SFU_0008_0008_Walking002_poses",
    #"0-HumanEva_S3_Walking_3_poses",
    #"0-BMLrub_rub096_0003_treadmill_jog_poses",
    #"0-HumanEva_S1_Jog_1_poses",
    #"0-DFaust_50007_50007_running_on_spot_poses",
    #"0-DFaust_50026_50026_running_on_spot_poses"
]

filtered_data = {key: data[key] for key in selected_names if key in data}

if not filtered_data:
    print("没有找到匹配的数据集，请检查数据集名称是否正确。")
else:
    
    os.makedirs(os.path.dirname(output_pkl_path), exist_ok=True)
    joblib.dump(filtered_data, output_pkl_path)
    print(f"已保存筛选后的数据到 {output_pkl_path}")
'''