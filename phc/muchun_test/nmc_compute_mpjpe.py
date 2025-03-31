import pickle
import joblib
import numpy as np
import os
from collections import defaultdict
from tqdm import tqdm
from scipy.spatial.transform import Rotation as sRot


def compute_error_vel(pred, gt):
    return np.linalg.norm(np.diff(pred, axis=0) - np.diff(gt, axis=0), axis=2)

def compute_error_accel(pred, gt):
    return np.linalg.norm(np.diff(pred, n=2, axis=0) - np.diff(gt, n=2, axis=0), axis=2)

def p_mpjpe(pred, gt):
    mu_pred = np.mean(pred, axis=1, keepdims=True)
    mu_gt = np.mean(gt, axis=1, keepdims=True)
    pred_centered = pred - mu_pred
    gt_centered = gt - mu_gt
    U, _, Vt = np.linalg.svd(np.einsum('ijk,ijl->ikl', pred_centered, gt_centered))
    R = np.einsum('ijk,ikl->ijl', U, Vt)
    aligned_pred = np.einsum('ijk,ikl->ijl', pred_centered, R) + mu_gt
    return np.linalg.norm(aligned_pred - gt, axis=2)

def compute_metrics_lite(pred_pos_all, gt_pos_all, pred_rot_all=None, gt_rot_all=None, root_idx=0):
    metrics = {}

    all_mpjpe_g = []
    all_mpjpe_l = []
    all_mpjpe_pa = []
    all_accel_dist = []
    all_vel_dist = []
    
    for idx in tqdm(range(len(pred_pos_all)), desc="Processing Metrics"):
        jpos_pred = pred_pos_all[idx].copy()
        jpos_gt = gt_pos_all[idx].copy()
        min_length = min(jpos_gt.shape[0], jpos_pred.shape[0])
        jpos_gt = jpos_gt[:min_length]
        jpos_pred = jpos_pred[:min_length]

        mpjpe_g = np.linalg.norm(jpos_gt - jpos_pred, axis=2) * 1000 
        vel_dist = compute_error_vel(jpos_pred, jpos_gt) * 1000 
        accel_dist = compute_error_accel(jpos_pred, jpos_gt) * 1000  
        jpos_pred -= jpos_pred[:, [root_idx]]
        jpos_gt -= jpos_gt[:, [root_idx]]

        pa_mpjpe = p_mpjpe(jpos_pred, jpos_gt) * 1000 
        mpjpe = np.linalg.norm(jpos_pred - jpos_gt, axis=2) * 1000  

        all_mpjpe_g.append(np.mean(mpjpe_g))  
        all_mpjpe_l.append(np.mean(mpjpe))
        all_mpjpe_pa.append(np.mean(pa_mpjpe))
        all_accel_dist.append(np.mean(accel_dist))
        all_vel_dist.append(np.mean(vel_dist))
    
    metrics["mpjpe_g"] = np.mean(all_mpjpe_g)  
    metrics["mpjpe_l"] = np.mean(all_mpjpe_l)
    metrics["mpjpe_pa"] = np.mean(all_mpjpe_pa)
    metrics["accel_dist"] = np.mean(all_accel_dist)
    metrics["vel_dist"] = np.mean(all_vel_dist)
    '''
    
    metrics["mpjpe_g"] = sum(all_mpjpe_g)  
    metrics["mpjpe_l"] = sum(all_mpjpe_l)
    metrics["mpjpe_pa"] = sum(all_mpjpe_pa)
    metrics["accel_dist"] = sum(all_accel_dist)
    metrics["vel_dist"] = sum(all_vel_dist)
'''

    return metrics


def load_pkl(filepath):
    with open(filepath, "rb") as f:
        return joblib.load(f)
def main():
    base_path = "/hdd/xinpeng/PHC_xinpeng/output/HumanoidIm/phc_comp_3/phc_act/amass_isaac_simple_run_upright_slim"
    output_dir = "/hdd/xinpeng/PHC"
    output_file = os.path.join(output_dir, "nmc_metrics_results.txt")  

    os.makedirs(output_dir, exist_ok=True)
   
    gt_path = os.path.join(base_path, "noise_False_0.05_4_0_0.1_egocentric_reference.pkl")
    gt_data = load_pkl(gt_path)
    gt_pos_all = gt_data["pred_pos"]
    gt_rot_all = gt_data["pred_rot"]
    

    keys = range(0, 501, 50)
    results = {}

    for key in keys:
        pred_path = os.path.join(base_path, f"noise_False_0.05_4_{key}_0.1_egocentric_reference.pkl")
        if not os.path.exists(pred_path):
            print(f"Warning: {pred_path} does not exist")
            continue
        
        pred_data = load_pkl(pred_path)
        pred_pos_all = pred_data["pred_pos"]
        pred_rot_all = pred_data["pred_rot"]

        metrics = compute_metrics_lite(pred_pos_all, gt_pos_all, pred_rot_all, gt_rot_all)
        results[key] = metrics

    with open(output_file, "a") as f:
        for key, metrics in results.items():
            f.write(f"Key: {key}\n")
            for metric_name, values in metrics.items():
                f.write(f"{metric_name}: {np.array2string(values, precision=4, separator=', ')}\n")
            f.write("\n")


    print(f"already saved to {output_file}")
if __name__ == "__main__":
    main()
