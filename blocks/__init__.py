from blocks.feedforward_block import FeedForwardBlock
from blocks.lowrank_bilinear_block import LowRankBilinearEncBlock, LowRankBilinearDecBlock

__factory = {
    'FeedForward': FeedForwardBlock,
    'LowRankBilinearEnc': LowRankBilinearEncBlock,
    'LowRankBilinearDec': LowRankBilinearDecBlock,
}

def names():
    return sorted(__factory.keys())

def create(name, *args, **kwargs):
    if name not in __factory:
        raise KeyError("Unknown blocks:", name)
    return __factory[name](*args, **kwargs)