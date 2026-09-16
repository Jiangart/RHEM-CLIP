device=2

LOG=${save_dir}"res.log"
echo ${LOG}
base_dir=${n_ctx[j]}_multiscale
CUDA_VISIBLE_DEVICES=${device} python trainconsistency.py --dataset mvtec --train_data_path /RHEMCLIP-main/dataset/mvtec_anomaly_detection \
--save_path ./checkpoints_new/${base_dir}/baseline \
--features_list 6 12 18 24 --image_size 518  --batch_size 8 --print_freq 1 \
--epoch 5 --save_freq 1  --n_ctx 12 \
--use_layer_specialized_prompt \
--consist_mode asymmetric \
--use_reliability_fusion


