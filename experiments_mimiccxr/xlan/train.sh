CUDA_VISIBLE_DEVICES=3,2,1,0 python3 -m torch.distributed.launch \
  --nproc_per_node=4 main.py --folder ./experiments_mimiccxr/xlan \
  --dataset_name MIMICCXR \
  --image_dir /content/mimic_cxr/images --ann_path /content/mimic_cxr/annotation.json