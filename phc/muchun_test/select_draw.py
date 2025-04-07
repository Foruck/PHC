import joblib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

pkl_file_path = '/hdd/xinpeng/PHC/data/amass/pkls/amass_isaac_im_train_take6_upright_slim.pkl'
data = joblib.load(pkl_file_path)

dataset_name = "0-SFU_0005_0005_Walking001_poses"               

if dataset_name not in data:
    print(f"Error: Dataset '{dataset_name}' not found in the pkl file!")
    exit()

dataset = data[dataset_name]

trans_orig = dataset.get('trans_orig', None)
root_trans_offset = dataset.get('root_trans_offset', None)

if trans_orig is None or root_trans_offset is None:
    print(f"No 'trans_orig' or 'root_trans_offset' data found in dataset {dataset_name}")
    exit()

num_frames, _ = trans_orig.shape
print(f"Shape of 'trans_orig': {trans_orig.shape}")
print(f"Shape of 'root_trans_offset': {root_trans_offset.shape}")

root_trans_offset = root_trans_offset.numpy()



# 计算骨盆的实际位置：骨盆位置 = trans_orig + root_trans_offset
pelvis_positions = trans_orig + root_trans_offset

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim([pelvis_positions[:, 0].min(), pelvis_positions[:, 0].max()])
ax.set_ylim([pelvis_positions[:, 1].min(), pelvis_positions[:, 1].max()])
ax.set_zlim([pelvis_positions[:, 2].min(), pelvis_positions[:, 2].max()])
ax.set_title("Pelvis Motion Trajectory")
sc = ax.scatter([], [], [], color='r', s=50)
line, = ax.plot([], [], [], color='b')

def update(frame):
   
    ax.cla()
    
    ax.set_xlim([pelvis_positions[:, 0].min(), pelvis_positions[:, 0].max()])
    ax.set_ylim([pelvis_positions[:, 1].min(), pelvis_positions[:, 1].max()])
    ax.set_zlim([pelvis_positions[:, 2].min(), pelvis_positions[:, 2].max()])
    ax.set_title(f"Frame {frame} - Pelvis Motion")
    
    ax.scatter(pelvis_positions[frame, 0], pelvis_positions[frame, 1], pelvis_positions[frame, 2], color='r', s=50)
    
    ax.plot(pelvis_positions[:frame+1, 0], pelvis_positions[:frame+1, 1], pelvis_positions[:frame+1, 2], color='b')

    return ax

ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=50, blit=False)

save_path = os.path.join(os.getcwd(), f"{dataset_name}_pelvis_motion_with_offset.gif")
ani.save(save_path, writer='pillow', fps=30)

print(f"Animation saved to {save_path}")
