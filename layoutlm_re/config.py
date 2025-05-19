import torch

# --- Model & Tokenizer ---
# MODEL_NAME = "microsoft/layoutlmv3-large" # Or base
MODEL_NAME = "microsoft/layoutlmv3-base"
CACHE_DIR = "./hf_cache" # Optional: Cache directory for Hugging Face models/tokenizers

# --- Data ---
# TODO: Adjust paths to your actual data directories
TRAIN_DATA_PATH = "data/train.jsonl" # Example format: json lines
VAL_DATA_PATH = "data/test.jsonl"
IMAGE_DIR = "/home/tamilselvan/Desktop/idp/dataset/FUNSD/" # Directory containing document images

MAX_LENGTH = 512 # Max sequence length for tokenizer
ENTITY_POOLING = "first" # How to get entity embeddings: "first" or "mean"
MAX_ENTITIES = 32 
ENTITY_INDEX_PAD_VALUE = -1 

# --- Training ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 100 # As an example
BATCH_SIZE = 2 # As mentioned in the paper
LEARNING_RATE = 5e-5
OPTIMIZER = "AdamW"
WEIGHT_DECAY = 0.01
GRAD_CLIP_NORM = 1.0 # Optional gradient clipping
ACCUMULATION_STEPS = 4 

# --- Output ---
OUTPUT_DIR = "./output"
CHECKPOINT_FILENAME = "best_model.pt"

# --- Enhancement Flags ---
# Set these flags to True/False based on the experiment you want to run
USE_IMAGE = True # LayoutLMv3 uses images by default

# Entity Type Information
USE_EEF = False # Joint EE and RE Fine-tuning
USE_EM = True # Entity Marker (prepend type to entity text)

# Geometry & Spatial Understanding
USE_BBO = True # Bounding Box Ordering (sort by vertical position)
USE_BBS = False # Bounding Box Shuffling (during training batches) - Requires custom collate_fn
USE_LC = False # Layout Concatenation (skip connection for coordinates)

# Post-processing / Advanced Loss
USE_RSF = True # Restriction on Selection of Fathers (post-processing)
RSF_TAU = 0.1 # Margin for RSF (example value)
USE_RSF_LOSS = False # Use variance loss during training (requires RSF implementation)

# --- EEF Specific (if USE_EEF = True) ---
# TODO: Define your entity labels and mapping
ENTITY_LABELS = ["O", "B-QUESTION", "I-QUESTION", "B-ANSWER", "I-ANSWER", "B-HEADER", "I-HEADER"] # Example from FUNSD
ID_TO_LABEL = {i: label for i, label in enumerate(ENTITY_LABELS)}
LABEL_TO_ID = {label: i for i, label in enumerate(ENTITY_LABELS)}
EE_LOSS_WEIGHT = 0.5 # Weight for EE loss in combined loss

# --- Logging ---
LOG_INTERVAL = 50 # Log training loss every N steps

# --- Miscellaneous ---
SEED = 42 # For reproducibility