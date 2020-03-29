from layers.low_rank import LowRank
from layers.basic_att import BasicAtt
from layers.sc_att import SCAtt

__factory = {
    'LowRank': LowRank,
    'BasicAtt': BasicAtt,
    'SCAtt': SCAtt,
}

def names():
    return sorted(__factory.keys())

def create(name, *args, **kwargs):
    if name not in __factory:
        raise KeyError("Unknown layer:", name)
    return __factory[name](*args, **kwargs)