#CUDA_VISIBLE_DEVICES=0  -m torch.distributed.launch --nproc_per_node=1
python3 main.py --folder ./artimes_experiments/iuxray_rgmg  --resume 0 --submodel rgmg --KG_path /project/CVML/pretrained_kg/rgmg_iuxray_pretrain.pth --dataset_name IUXRAY --image_dir /project/CVML/Parallel-R2Gen-KG/data/iu_xray/images/ --ann_path /project/CVML/Parallel-R2Gen-KG/data/iuxray/annotation.json


###if you want to use checkpoint, download your model in experiments_mimiccxr/xtransformer/snapshot and change 0 to your model's number
