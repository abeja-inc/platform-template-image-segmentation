# coding: utf-8
# ML parameters.


import os


# Train params
BATCH_SIZE = int(os.environ.get('BATCH_SIZE', '32'))
EPOCHS = int(os.environ.get('EPOCHS', '50'))
EARLY_STOPPING_TEST_SIZE = float(os.environ.get('EARLY_STOPPING_TEST_SIZE', '0.2'))
LEARNING_RATE = float(os.environ.get('LEARNING_RATE', '0.01'))
MOMENTUM = float(os.environ.get('MOMENTUM','0.9'))
WEIGHT_DECAY = float(os.environ.get('WEIGHT_DECAY','1e-4'))
USE_CACHE = bool(os.environ.get('USE_CACHE', 'True').lower() == 'true')
USE_ON_MEMORY = bool(os.environ.get('USE_ON_MEMORY', 'False').lower() == 'true')
if USE_ON_MEMORY:
    USE_CACHE = True
NUM_DATA_LOAD_THREAD = int(os.environ.get('NUM_DATA_LOAD_THREAD', '1'))
if NUM_DATA_LOAD_THREAD > BATCH_SIZE:
    NUM_DATA_LOAD_THREAD = BATCH_SIZE

# fcn_resnet101, deeplabv3_resnet101
SEG_MODEL = os.environ.get('SEG_MODEL', 'deeplabv3_resnet101')
DEVICE = os.environ.get('DEVICE','cuda')
FINE_TUNING = bool(os.environ.get('FINE_TUNING', 'False').lower() == 'true')
PRINT_FREQ = int(os.environ.get('PRINT_FREQ', '10'))
RESUME = os.environ.get('RESUME', '')
AUX_LOSS = bool(os.environ.get('AUX_LOSS','False').lower() == 'true')
TEST_ONLY = bool(os.environ.get('TEST_ONLY', 'False').lower() == 'true')
PRETRAINED = bool(os.environ.get('PRETRAINED', 'True').lower() == 'true')
#OUTPUT_DIR = os.environ.get('OUTPUT_DIR','.')

# distributed training parameters
DISTRIBUTED = False
WORLD_SIZE = 1
RANK = 1
DIST_URL=os.environ.get("DIST_URL", "env://")
GPU = 0
DIST_BACKEND = ''

# For print
parameters = {
    'BATCH_SIZE': BATCH_SIZE,
    'EPOCHS': EPOCHS,
    'EARLY_STOPPING_TEST_SIZE': EARLY_STOPPING_TEST_SIZE,
    'LEARNING_RATE': LEARNING_RATE,
    'MOMENTUM': MOMENTUM,
    'USE_CACHE': USE_CACHE,
    'USE_ON_MEMORY': USE_ON_MEMORY,
    'SEG_MODEL': SEG_MODEL,
    'DEVICE': DEVICE,
    'FINE_TUNING':FINE_TUNING,
    'PRETRAINED':PRETRAINED,
    'WEIGHT_DECAY':WEIGHT_DECAY,
    'USE_CACHE':USE_CACHE,
    'USE_ON_MEMORY':USE_ON_MEMORY,
    'NUM_DATA_LOAD_THREAD':NUM_DATA_LOAD_THREAD,
    'PRINT_FREQ':PRINT_FREQ,
    'AUX_LOSS':AUX_LOSS,
    'TEST_ONLY':TEST_ONLY,
    'DISTRIBUTED':DISTRIBUTED,
    'WORLD_SIZE':WORLD_SIZE,
    'DIST_URL':DIST_URL,
    'RANK':RANK,
    'GPU':GPU,
    'DIST_BACKEND':DIST_BACKEND
}

