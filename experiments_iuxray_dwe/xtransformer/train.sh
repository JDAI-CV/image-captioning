CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch --nproc_per_node=1 main.py --folder ./experiments_iuxray_dwe/xtransformer  --resume 0 --encoder_mode dualwayencoder --dataset_name IUXRAY --submodel VSEGCN

###if you want to use checkpoint, download your model in experiments_mimiccxr/xtransformer/snapshot and change 0 to your model's number
