import os
import os.path as osp
import numpy as np

from easydict import EasyDict as edict

__C = edict()
# Consumers can get config by:
#   from lib.config import cfg
cfg = __C

# ---------------------------------------------------------------------------- #
# Training options
# ---------------------------------------------------------------------------- #
__C.TRAIN = edict()

# Minibatch size
__C.TRAIN.BATCH_SIZE = 10

# scheduled sampling
__C.TRAIN.SCHEDULED_SAMPLING = edict()

__C.TRAIN.SCHEDULED_SAMPLING.START = 0

__C.TRAIN.SCHEDULED_SAMPLING.INC_EVERY = 5

__C.TRAIN.SCHEDULED_SAMPLING.INC_PROB = 0.05

__C.TRAIN.SCHEDULED_SAMPLING.MAX_PROB = 0.25

# reinforcement learning
__C.TRAIN.REINFORCEMENT = edict()

__C.TRAIN.REINFORCEMENT.START = 30

# ---------------------------------------------------------------------------- #
# Inference ('test') options
# ---------------------------------------------------------------------------- #
__C.TEST = edict()

# Minibatch size
__C.TEST.BATCH_SIZE = 36


# ---------------------------------------------------------------------------- #
# Data loader options
# ---------------------------------------------------------------------------- #
__C.DATA_LOADER = edict()

# Data directory
__C.DATA_LOADER.NUM_WORKERS = 4

__C.DATA_LOADER.PIN_MEMORY = True

__C.DATA_LOADER.DROP_LAST = True

__C.DATA_LOADER.SHUFFLE = True

__C.DATA_LOADER.TRAIN_GV_FEAT = ''

__C.DATA_LOADER.TRAIN_ATT_FEATS = 'up_down_10_100'

__C.DATA_LOADER.VAL_GV_FEAT = ''

__C.DATA_LOADER.VAL_ATT_FEATS = 'up_down_10_100'

__C.DATA_LOADER.TEST_GV_FEAT = ''

__C.DATA_LOADER.TEST_ATT_FEATS = 'up_down_10_100'

__C.DATA_LOADER.TRAIN_ID = 'coco_train_image_id.txt'

__C.DATA_LOADER.VAL_ID = 'coco_val_image_id.txt'

__C.DATA_LOADER.TEST_ID = 'coco_test_image_id.txt'

__C.DATA_LOADER.INPUT_SEQ_PATH = 'coco_train_input.pkl'

__C.DATA_LOADER.TARGET_SEQ_PATH = 'coco_train_target.pkl'

__C.DATA_LOADER.SEQ_PER_IMG = 5

__C.DATA_LOADER.MAX_FEAT = -1

# ---------------------------------------------------------------------------- #
# Model options
# ---------------------------------------------------------------------------- #
__C.MODEL = edict()

__C.MODEL.PretrainedImageModel = '/content/image-captioning/model_auc14.dict.gz' # TODO # Modified

__C.MODEL.TYPE = 'UpDown'               # 'UpDown', 'XLAN', 'XTransformer'

__C.MODEL.SEQ_LEN = 60                  # include <EOS>/<BOS> # modified

__C.MODEL.VOCAB_SIZE = 760             # exclude <EOS>/<BOS> # TODO : IUXRAY: 760

__C.MODEL.WORD_EMBED_DIM = 512         # TODO # Modified

__C.MODEL.WORD_EMBED_ACT = 'NONE'       # 'RELU', 'CELU', 'NONE'

__C.MODEL.WORD_EMBED_NORM = False

__C.MODEL.DROPOUT_WORD_EMBED = 0.0

__C.MODEL.GVFEAT_DIM = 2048 # TODO

__C.MODEL.GVFEAT_EMBED_DIM = -1

__C.MODEL.GVFEAT_EMBED_ACT = 'NONE'     # 'RELU', 'CELU', 'NONE'

__C.MODEL.DROPOUT_GV_EMBED = 0.0

__C.MODEL.ATT_FEATS_DIM = 1024  # Not used. Modified on the init stage of model

__C.MODEL.ATT_FEATS_EMBED_DIM = -1

__C.MODEL.ATT_FEATS_EMBED_ACT = 'NONE'   # 'RELU', 'CELU', 'NONE'

__C.MODEL.DROPOUT_ATT_EMBED = 0.0

__C.MODEL.ATT_FEATS_NORM = False

__C.MODEL.ATT_HIDDEN_SIZE = 512 # TODO

__C.MODEL.ATT_HIDDEN_DROP = 0.0

__C.MODEL.ATT_ACT = 'RELU'  # 'RELU', 'CELU', 'TANH'

__C.MODEL.RNN_SIZE = 1000 # TODO

__C.MODEL.DROPOUT_LM = 0.5

# BOTTOM_UP
__C.MODEL.BOTTOM_UP = edict()

__C.MODEL.BOTTOM_UP.DROPOUT_FIRST_INPUT = 0.0

__C.MODEL.BOTTOM_UP.DROPOUT_SEC_INPUT = 0.0

# Transformer
__C.MODEL.TRANSFORMER = edict()

__C.MODEL.TRANSFORMER.PE_MAX_LEN = 5000


# Bilinear
__C.MODEL.BILINEAR = edict()

__C.MODEL.BILINEAR.DIM = -1

__C.MODEL.BILINEAR.ENCODE_ATT_MID_DIM = [1]

__C.MODEL.BILINEAR.DECODE_ATT_MID_DIM = [1]

__C.MODEL.BILINEAR.ENCODE_ATT_MID_DROPOUT = 0.0

__C.MODEL.BILINEAR.DECODE_ATT_MID_DROPOUT = 0.0

__C.MODEL.BILINEAR.ATT_DIM = 1000

__C.MODEL.BILINEAR.ACT = 'RELU'  # 'RELU', 'CELU', 'TANH', 'GLU'

__C.MODEL.BILINEAR.ENCODE_DROPOUT = 0.1

__C.MODEL.BILINEAR.DECODE_DROPOUT = 0.1

__C.MODEL.BILINEAR.ENCODE_LAYERS = 1

__C.MODEL.BILINEAR.DECODE_LAYERS = 1

__C.MODEL.BILINEAR.TYPE = 'LowRank'

__C.MODEL.BILINEAR.ATTTYPE = 'SCAtt'

__C.MODEL.BILINEAR.HEAD = 8

__C.MODEL.BILINEAR.ENCODE_FF_DROPOUT = 0.1

__C.MODEL.BILINEAR.DECODE_FF_DROPOUT = 0.1

__C.MODEL.BILINEAR.ENCODE_BLOCK = 'LowRankBilinearEnc'

__C.MODEL.BILINEAR.DECODE_BLOCK = 'LowRankBilinearDec'

__C.MODEL.BILINEAR.ELU_ALPHA = 1.0

__C.MODEL.BILINEAR.BIFEAT_EMB_ACT = 'RELU'

__C.MODEL.BILINEAR.ENCODE_BIFEAT_EMB_DROPOUT = 0.3

__C.MODEL.BILINEAR.DECODE_BIFEAT_EMB_DROPOUT = 0.3

# ---------------------------------------------------------------------------- #
# Solver options
# ---------------------------------------------------------------------------- #
__C.SOLVER = edict()

# Base learning rate for the specified schedule
__C.SOLVER.BASE_LR = 0.0005

# Solver type
__C.SOLVER.TYPE = 'ADAM'                 # 'ADAM', 'ADAMAX', 'SGD', 'ADAGRAD', 'RMSPROP', 'RADAM'

# Maximum number of SGD iterations
__C.SOLVER.MAX_EPOCH = 30

__C.SOLVER.MAX_ITER = 60000

__C.SOLVER.GRAD_CLIP = 0.1               # Norm:0.5 , Clamp:0.1

__C.SOLVER.GRAD_CLIP_TYPE = 'Clamp'      # 'Clamp', 'Norm'

# L2 regularization hyperparameter
__C.SOLVER.WEIGHT_DECAY = 0.0005

__C.SOLVER.WEIGHT_DECAY_BIAS = 0.0

__C.SOLVER.BIAS_LR_FACTOR = 2

__C.SOLVER.DISPLAY = 100

__C.SOLVER.TEST_INTERVAL = 1

__C.SOLVER.SNAPSHOT_ITERS = 3

# SGD
__C.SOLVER.SGD = edict()
__C.SOLVER.SGD.MOMENTUM = 0.9

# ADAM
__C.SOLVER.ADAM = edict()
__C.SOLVER.ADAM.BETAS = [0.9, 0.999]
__C.SOLVER.ADAM.EPS = 1e-8

# LR_POLICY
# Schedule type (see functions in utils.lr_policy for options)
# E.g., 'step', 'steps_with_decay', ...
__C.SOLVER.LR_POLICY = edict()
__C.SOLVER.LR_POLICY.TYPE = 'Step'       # 'Fix', 'Step', 'Noam', 'Plateau'
__C.SOLVER.LR_POLICY.GAMMA = 0.8         # For 'step', the current LR is multiplied by SOLVER.GAMMA at each step
__C.SOLVER.LR_POLICY.STEP_SIZE = 3       # Uniform step size for 'steps' policy
__C.SOLVER.LR_POLICY.STEPS = (3,)        # Non-uniform step iterations for 'steps_with_decay' or 'steps_with_lrs' policies
__C.SOLVER.LR_POLICY.SETP_TYPE = 'Epoch' # 'Epoch', 'Iter'

__C.SOLVER.LR_POLICY.WARMUP = 20000      # For Noam only
__C.SOLVER.LR_POLICY.FACTOR = 1.0        # For Noam only
__C.SOLVER.LR_POLICY.MODEL_SIZE = 1024   # For Noam only

__C.SOLVER.LR_POLICY.PLATEAU_FACTOR = 0.5
__C.SOLVER.LR_POLICY.PLATEAU_PATIENCE = 3

# ---------------------------------------------------------------------------- #
# Losses options
# ---------------------------------------------------------------------------- #
__C.LOSSES = edict()

__C.LOSSES.XE_TYPE = 'CrossEntropy'      # 'CrossEntropy', 'LabelSmoothing'

__C.LOSSES.RL_TYPE = 'RewardCriterion'

__C.LOSSES.LABELSMOOTHING = 0.0

# ---------------------------------------------------------------------------- #
# SCORER options
# ---------------------------------------------------------------------------- #
__C.SCORER = edict()

__C.SCORER.TYPES = ['Cider']

__C.SCORER.WEIGHTS = [1.0]

__C.SCORER.GT_PATH = 'coco_train_gts.pkl'

__C.SCORER.CIDER_CACHED = 'coco_train_cider.pkl'

# ---------------------------------------------------------------------------- #
# PARAM options
# ---------------------------------------------------------------------------- #
__C.PARAM = edict()

__C.PARAM.WT = 'WT'

__C.PARAM.GLOBAL_FEAT = 'GV_FEAT'

__C.PARAM.ATT_FEATS = 'ATT_FEATS'

__C.PARAM.ATT_FEATS_MASK = 'ATT_FEATS_MASK'

__C.PARAM.P_ATT_FEATS = 'P_ATT_FEATS'

__C.PARAM.STATE = 'STATE'

__C.PARAM.INPUT_SENT = 'INPUT_SENT'

__C.PARAM.TARGET_SENT = 'TARGET_SENT'

__C.PARAM.INDICES = 'INDICES'

# ---------------------------------------------------------------------------- #
# Inference options
# ---------------------------------------------------------------------------- #
__C.INFERENCE = edict()

__C.INFERENCE.VOCAB = 'coco_vocabulary.txt'

__C.INFERENCE.ID_KEY = 'image_id'

__C.INFERENCE.CAP_KEY = 'report' # Modified

__C.INFERENCE.EVAL = 'COCO'

__C.INFERENCE.VAL_ANNFILE = 'captions_val5k.json'

__C.INFERENCE.TEST_ANNFILE = 'captions_test5k.json'

__C.INFERENCE.BEAM_SIZE = 1

__C.INFERENCE.GREEDY_DECODE = True # Greedy decode or sample decode

__C.INFERENCE.COCO_PATH = '../coco_caption'

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
# A small number that's used many times
__C.EPS = 1e-14

# Root directory of project
__C.ROOT_DIR = os.getcwd()

# Logger name
__C.LOGGER_NAME = 'log'

# Image Mean
__C.MEAN = [0.485, 0.456, 0.406]

# Image std
__C.STD = [0.229, 0.224, 0.225]

__C.SEED = -1.0

__C.TEMP_DIR = './data/temp'

def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    #for k, v in a.iteritems(): python2
    for k, v in a.items(): # python3
        # a must specify keys that are in b
        #if not b.has_key(k):
        if not k in b:
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                'for config key: {}').format(type(b[k]),
                                                            type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v

def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f))

    _merge_a_into_b(yaml_cfg, __C)