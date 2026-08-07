import torch

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64
INT_DTYPE = torch.int64

Q = 12289
N_LAT = 256
M_LAT = 512
S_LAT = 64
D1 = 128
D2 = 256
D_REPR = D1 + D2
KAPPA = 128

GAUSSIAN_SIGMA = 3.2

ALPHA_RENYI = 2.0
THETA = 0.5
EPSILON_0 = 1.0
DELTA = 1e-5
LAMBDA_PARAM = 2.0
KNN_K = 5
PROJ_DIM = 16
NOISE_K = 8.0

NUM_PARTICIPANTS = 10
NUM_ROUNDS = 20
LOCAL_EPOCHS = 2
LOCAL_LR = 0.01
BATCH_SIZE = 128
DIRICHLET_ALPHA = 0.5

NUM_CLASSES = 10
DATA_DIR = "./data"

USE_AMP = True
NUM_WORKERS = 2
PREFETCH_FACTOR = 4
EVAL_BATCH_MULT = 2
USE_CHANNELS_LAST = True
USE_COMPILE = False


DATASETS = {
    "cifar10": {
        "channels": 3,
        "image_size": 32,
        "num_classes": 10,
        "rounds": 20,
        "mean": (0.4914, 0.4822, 0.4465),
        "std": (0.2023, 0.1994, 0.2010),
        "augment": True,
        "label": "CIFAR-10",
    },
    "mnist": {
        "channels": 1,
        "image_size": 28,
        "num_classes": 10,
        "rounds": 15,
        "mean": (0.1307,),
        "std": (0.3081,),
        "augment": False,
        "label": "MNIST",
    },
    "fashion_mnist": {
        "channels": 1,
        "image_size": 28,
        "num_classes": 10,
        "rounds": 15,
        "mean": (0.2860,),
        "std": (0.3530,),
        "augment": False,
        "label": "Fashion-MNIST",
    },
}

SEED = 42

MAX_ENHANCEMENT_ITERS = 10
CONVERGENCE_TOL = 1e-4

MERKLE_BLOCK_SIZE = 256
