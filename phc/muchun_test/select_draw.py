import joblib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

# 读取 pkl 文件
pkl_file_path = '/hdd/xinpeng/PHC/data/amass/pkls/amass_isaac_im_train_take6_upright_slim.pkl'
data = joblib.load(pkl_file_path)

dataset_name = "0-TotalCapture_s5_walking2_poses"               

if dataset_name not in data:
    print(f"Error: Dataset '{dataset_name}' not found in the pkl file!")
    exit()

dataset = data[dataset_name]

# 获取骨盆的平移信息
trans_orig = dataset.get('trans_orig', None)
if trans_orig is None:
    print(f"No 'trans_orig' data found in dataset {dataset_name}")
    exit()

num_frames, _ = trans_orig.shape
print(f"Shape of 'trans_orig': {trans_orig.shape}")

# 设置绘图
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim([trans_orig[:, 0].min(), trans_orig[:, 0].max()])
ax.set_ylim([trans_orig[:, 1].min(), trans_orig[:, 1].max()])
ax.set_zlim([trans_orig[:, 2].min(), trans_orig[:, 2].max()])
ax.set_title("Pelvis Motion Trajectory")
sc = ax.scatter([], [], [], color='r', s=50)
line, = ax.plot([], [], [], color='b')

# 更新动画帧
def update(frame):
    ax.cla()
    ax.set_xlim([trans_orig[:, 0].min(), trans_orig[:, 0].max()])
    ax.set_ylim([trans_orig[:, 1].min(), trans_orig[:, 1].max()])
    ax.set_zlim([trans_orig[:, 2].min(), trans_orig[:, 2].max()])
    ax.set_title(f"Frame {frame} - Pelvis Motion")
    
    ax.scatter(trans_orig[frame, 0], trans_orig[frame, 1], trans_orig[frame, 2], color='r', s=50)
    ax.plot(trans_orig[:frame+1, 0], trans_orig[:frame+1, 1], trans_orig[:frame+1, 2], color='b')
    
    return ax

# 创建动画
ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=50, blit=False)

# 保存动画
save_path = os.path.join(os.getcwd(), f"{dataset_name}_pelvis_motion.gif")
ani.save(save_path, writer='pillow', fps=30)

print(f"Animation saved to {save_path}")
