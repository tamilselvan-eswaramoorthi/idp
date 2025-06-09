import torch

# --- Model & Tokenizer ---
MODEL_NAME = "microsoft/layoutlmv3-base" # "microsoft/layoutlmv3-base"
IMAGE_TOKENIZER_NAME = "microsoft/dit-base"  # Using DIT for image tokenization
CACHE_DIR = "./hf_cache"
OUTPUT_DIR = "./output" 
CHECKPOINT_FILENAME = 'best_model.pt'

# --- Data ---
TRAIN_DATA_PATH = "funsd_dataset/training_data/data.jsonl"
VAL_DATA_PATH = "funsd_dataset/testing_data/data.jsonl" 
TRAIN_PATH = "funsd_dataset/training_data/"
VAL_PATH = "funsd_dataset/testing_data/"

# Original LayoutLM uses 512 tokens, but can try 1024 if document is larger
MAX_LENGTH = 512
ENTITY_POOLING = "first"
MAX_ENTITIES = 32 
ENTITY_INDEX_PAD_VALUE = -1 

# --- Training ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 100
BATCH_SIZE = 2
LEARNING_RATE = 5e-5
OPTIMIZER = "AdamW"
WEIGHT_DECAY = 0.01
GRAD_CLIP_NORM = 1.0
ACCUMULATION_STEPS = 4 # Effective batch size = BATCH_SIZE * ACCUMULATION_STEPS

WARMUP_RATIO = 0.1  # Warm up learning rate for this portion of training
PREDICTION_THRESHOLD = 0.4  # Slightly lower threshold can improve recall
ALLOW_SELF_RELATIONS = False
USE_AMP_INFERENCE = False
NUM_WORKERS = 4  # For faster data loading

# Post-processing & Loss
USE_RSF = True
RSF_TAU = 0.2  # Increased from 0.1 for stronger filtering
USE_RSF_LOSS = True  # Changed to True - helps train with the RSF constraint
RSF_LOSS_WEIGHT = 0.05  # Reduced to prevent instability
LOG_NAN_WARNINGS = True  # Add a flag to control NaN warning verbosity

# Add a gradient clipping parameter to help prevent gradient explosion
GRAD_CLIP_VALUE = 1.0

# --- EEF Specific ---
ENTITY_LABELS = ["O", "B-QUESTION", "I-QUESTION", "B-ANSWER", "I-ANSWER", "B-HEADER", "I-HEADER"]
ID_TO_LABEL = {i: label for i, label in enumerate(ENTITY_LABELS)}
LABEL_TO_ID = {label: i for i, label in enumerate(ENTITY_LABELS)}

# --- Logging ---
LOG_INTERVAL = 150  # More frequent logging

# --- Miscellaneous ---
SEED = 42