# coding: utf-8
# ML parameters.


import os

import torch


# Train params
BATCH_SIZE = int(os.environ.get('BATCH_SIZE', '32'))
EPOCHS = int(os.environ.get('EPOCHS', '50'))
EARLY_STOPPING_TEST_SIZE = float(os.environ.get('EARLY_STOPPING_TEST_SIZE', '0.2'))
LEARNING_RATE = float(os.environ.get('LEARNING_RATE', '0.01'))
MOMENTUM = float(os.environ.get('MOMENTUM','0.9'))
WEIGHT_DECAY = float(os.environ.get('WEIGHT_DECAY','1e-4'))
USE_CACHE = bool(os.environ.get('USE_CACHE', 'True').lower() == 'true')
USE_ON_MEMORY = bool(os.environ.get('USE_ON_MEMORY', 'True').lower() == 'true')
if USE_ON_MEMORY:
    USE_CACHE = True
NUM_DATA_LOAD_THREAD = int(os.environ.get('NUM_DATA_LOAD_THREAD', '0'))
if NUM_DATA_LOAD_THREAD > BATCH_SIZE:
    NUM_DATA_LOAD_THREAD = BATCH_SIZE
RANDOM_SEED = int(os.environ.get('RANDOM_SEED', '42'))

# fcn_resnet101, deeplabv3_resnet101
SEGMENTATION_MODEL = os.environ.get('SEGMENTATION_MODEL', 'deeplabv3_resnet101')
DEVICE = os.environ.get('DEVICE','cuda')
FINE_TUNING = bool(os.environ.get('FINE_TUNING', 'False').lower() == 'true')
PRINT_FREQ = int(os.environ.get('PRINT_FREQ', '10'))
RESUME = os.environ.get('RESUME', '')
AUX_LOSS = bool(os.environ.get('AUX_LOSS','False').lower() == 'true')
PRETRAINED = bool(os.environ.get('PRETRAINED', 'True').lower() == 'true')

# distributed training parameters
DISTRIBUTED = True
WORLD_SIZE = 1
RANK = 1
DIST_URL=os.environ.get("DIST_URL", "env://")
GPU = 0
DIST_BACKEND = ''
if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
    RANK = int(os.environ["RANK"])
    WORLD_SIZE = int(os.environ['WORLD_SIZE'])
    GPU = int(os.environ['LOCAL_RANK'])
    DIST_BACKEND = 'nccl'
elif 'SLURM_PROCID' in os.environ:
    RANK = int(os.environ['SLURM_PROCID'])
    GPU = RANK % torch.cuda.device_count()
    DIST_BACKEND = 'nccl'
else:
    DISTRIBUTED = False

# For print
parameters = {
    'BATCH_SIZE': BATCH_SIZE,
    'EPOCHS': EPOCHS,
    'EARLY_STOPPING_TEST_SIZE': EARLY_STOPPING_TEST_SIZE,
    'LEARNING_RATE': LEARNING_RATE,
    'MOMENTUM': MOMENTUM,
    'SEGMENTATION_MODEL': SEGMENTATION_MODEL,
    'DEVICE': DEVICE,
    'FINE_TUNING': FINE_TUNING,
    'PRETRAINED': PRETRAINED,
    'WEIGHT_DECAY': WEIGHT_DECAY,
    'USE_CACHE': USE_CACHE,
    'USE_ON_MEMORY': USE_ON_MEMORY,
    'NUM_DATA_LOAD_THREAD': NUM_DATA_LOAD_THREAD,
    'RANDOM_SEED': RANDOM_SEED,
    'PRINT_FREQ': PRINT_FREQ,
    'AUX_LOSS': AUX_LOSS,
    'DISTRIBUTED': DISTRIBUTED,
    'WORLD_SIZE': WORLD_SIZE,
    'DIST_URL': DIST_URL,
    'RANK': RANK,
    'GPU': GPU,
    'DIST_BACKEND': DIST_BACKEND
}

