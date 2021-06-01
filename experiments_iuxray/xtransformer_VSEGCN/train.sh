CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch --nproc_per_node=1 main.py --folder ./experiments_iuxray/xtransformer_VSEGCN --submodel VSEGCN \
