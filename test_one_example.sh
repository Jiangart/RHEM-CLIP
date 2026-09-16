
device=2


CUDA_VISIBLE_DEVICES=2 python test_one_example.py \
--image_path "/data4/~_project/Dataset/000.png" \
--checkpoint_path "/data4/~_project/RHEMCLIP2/RHEMCLIP-main/checkpoints1/checkpoints9156/0_12_0_multiscale/yuan/consistency/1_alpha0.15beta1/epoch_5.pth" \
--save_path "./vis_results"
