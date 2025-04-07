import os

reaction_idx_list= [23]
amplitude_list= [500]

for reaction_idx in reaction_idx_list:
    for amplitude in amplitude_list:
        
        print(f"Running experiment with reaction_idx={reaction_idx} and amplitude={amplitude}...")

        command = f"""
        CUDA_VISIBLE_DEVICES=7 python phc/run_hydra.py learning=im_pnn_big env=env_im_pnn robot=smpl_humanoid env.motion_file=/hdd/xinpeng/PHC/phc/muchun_test/selected_motion/0-HumanEva_S3_Walking_3_poses.pkl exp_name=fatigue env.training_prim=0 +env.pd_modifier=True env.num_envs=1536 learning.params.config.save_frequency=5000 +env.MFObs=True +env.use_fatigue=True +env.peak_path=output/HumanoidIm/pd_modifier/torque_limits.pkl +env.randomized_fatigue=False env.models=\['output/HumanoidIm/fatigue/Humanoid_00035000.pth'\] epoch=35000 test=True im_eval=True +env.reaction_idx={reaction_idx} +env.reaction_mode=egocentric_reference +env.external_interference_amplitude={amplitude}

        """
        os.system(command)
        print(f"Experiment with reaction_idx={reaction_idx} and amplitude={amplitude} completed.\n")
