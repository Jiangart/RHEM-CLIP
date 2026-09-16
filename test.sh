
device=2

LOG=${save_dir}"res.log"
echo ${LOG} 
CUDA_VISIBLE_DEVICES=${device} python testconsistency.py --dataset visa  \
--data_path /data4/~_project/Dataset/visa \
--checkpoint_path ./RHEMCLIP-main/checkpoints_new/0_12_0_multiscale/baseline2/epoch_5.pth \
--features_list 6 12 18 24 --image_size 518 --n_ctx 12 \
--use_layer_specialized_prompt \
--use_reliability_fusion \
--enable_smoothing\
--enable_gradient

