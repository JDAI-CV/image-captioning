from models.updown import UpDown
from models.xlan import XLAN
from models.xtransformer import XTransformer
from models.dwextransformer import DWEXTransformer


__factory = {
    'UpDown': UpDown,
    'XLAN': XLAN,
    'XTransformer': XTransformer,
    'DWEXtransformer' : DWEXTransformer
}

def names():
    return sorted(__factory.keys())

def create(name, args, submodel, **kwargs):
    if name == 'XTransformer' and args.encoder_mode == 'dualwayencoder':
        name = 'DWEXtransformer'
    if name not in __factory:
        raise KeyError("Unknown caption model:", name)
    return __factory[name](args, submodel, **kwargs)
