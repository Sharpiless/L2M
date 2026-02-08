# CUDA_VISIBLE_DEVICES=5 python test_relative_homo_mmim.py \
#     --method l2mpp --choose_model 0 --save_dir saves \

# CUDA_VISIBLE_DEVICES=5 python test_relative_homo_mmim.py \
#     --method l2mpp --choose_model 1 --save_dir saves \

# CUDA_VISIBLE_DEVICES=5 python test_relative_homo_mmim.py \
    # --method l2mpp --choose_model 2 --save_dir saves \

# CUDA_VISIBLE_DEVICES=5 python test_relative_homo_event.py --method l2mpp --save_dir saves \

# CUDA_VISIBLE_DEVICES=5 python test_relative_homo_depth.py --method l2mpp --save_dir saves

# CUDA_VISIBLE_DEVICES=5 python test_relative_homo_anhir.py --method l2mpp --save_dir saves

# CUDA_VISIBLE_DEVICES=5 python test_relative_homo_cima.py --method l2mpp --save_dir saves

# CUDA_VISIBLE_DEVICES=5 python test_relative_pose_weather.py --method l2mpp --save_dir saves

# CUDA_VISIBLE_DEVICES=5 python test_relative_pose_weather_v2.py --method l2mpp --save_dir saves

CUDA_VISIBLE_DEVICES=5 python test_relative_pose_waternerf.py --method l2mpp --save_dir saves

    # --ckpt /data6/liangyingping/git-code/gim/weights/l2mpp_mixed0208.ckpt