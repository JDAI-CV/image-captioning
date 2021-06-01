CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch \
  --nproc_per_node=1 main.py --folder ./experiments_mimiccxr/xtransformer \
  --dataset_name MIMICCXR \
  --image_dir /content/mimic_cxr/images --ann_path /content/mimic_cxr/annotation.json \
  --submodel VSEGCN 
