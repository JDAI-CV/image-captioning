CUDA_VISIBLE_DEVICES=0 python3 main.py \
  --folder ./experiments_mimiccxr/xlan_rl --resume 47 \
  --dataset_name MIMICCXR \
  --image_dir /content/mimic_cxr/images --ann_path /content/mimic_cxr/annotation.json
# 47 is the epoch number of the pretrained model
