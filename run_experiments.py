import os

reaction_idx_list = [0]
amplitude_list = [300]

for reaction_idx in reaction_idx_list:
    for amplitude in amplitude_list:
        print(f"Running experiment with reaction_idx={reaction_idx} and amplitude={amplitude}...")
        command = f"""
        CUDA_VISIBLE_DEVICES=4 python phc/run_hydra.py \\
            learning=im_mcp_big \\
            exp_name=phc_comp_3 \\
            env=env_im_getup_mcp_reaction \\
            robot=smpl_humanoid \\
            env.zero_out_far=False \\
            robot.real_weight_porpotion_boxes=False \\
            env.num_prim=3 \\
            env.motion_file=/hdd/caizixuan/PULSE/sample_data/amass_isaac_simple_run_upright_slim.pkl \\
            env.models=['output/HumanoidIm/phc_3/Humanoid.pth'] \\
            env.num_envs=256 \\
            headless=True \\
            epoch=-1 \\
            test=True \\
            im_eval=True \\
            collect_dataset=True \\
            env.reaction_idx={reaction_idx} \\
            env.reaction_mode=momentum \\
            env.external_interference_amplitude={amplitude}
        """
        os.system(command)
        print(f"Experiment with reaction_idx={reaction_idx} and amplitude-y={amplitude} completed.\n")
