#!/usr/bin/env python
# coding: utf-8

# # Training Phi-3-mini-128k-instruct to Learn Swift Programming Language
# 
# This notebook trains Microsoft's Phi-3-mini-128k-instruct model to understand and work with Swift code using a dataset of real Swift files.

# Install required libraries
# !pip install transformers datasets evaluate torch scikit-learn tqdm dropbox requests accelerate peft bitsandbytes aqlm
# Set PyTorch memory management environment variables to avoid fragmentation
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # Explicitly set to use 2 GPUs

# Import required libraries
import torch
import numpy as np
import random
import time
import collections
import psutil
import os
import gc
import json
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer, 
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from aqlm import AqlmConfig

# Define memory cleanup function
def cleanup_memory():
    """Clean up GPU memory to avoid fragmentation."""
    print("Cleaning up memory...")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
# Define resource monitoring function
def monitor_resources():
    """Monitor system and GPU resources."""
    # Monitor CPU and RAM
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    print(f"CPU memory usage: {memory_info.rss / 1024 / 1024:.2f} MB")
    
    # Monitor GPU if available
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"Number of GPUs: {num_gpus}")
        
        for i in range(num_gpus):
            if hasattr(torch.cuda, 'memory_allocated'):
                print(f"GPU {i} ({torch.cuda.get_device_name(i)})")
                print(f"  Memory allocated: {torch.cuda.memory_allocated(i) / (1024**3):.2f} GB")
                print(f"  Memory reserved: {torch.cuda.memory_reserved(i) / (1024**3):.2f} GB")
                if hasattr(torch.cuda, 'memory_stats'):
                    stats = torch.cuda.memory_stats(i)
                    if 'active_bytes.all.current' in stats:
                        print(f"  Active memory: {stats['active_bytes.all.current'] / (1024**3):.2f} GB")
                    if 'reserved_bytes.all.current' in stats:
                        print(f"  Reserved memory: {stats['reserved_bytes.all.current'] / (1024**3):.2f} GB")

# Check if GPU is available and configure for multi-GPU training
if torch.cuda.is_available():
    # Set up for distributed training on multiple GPUs
    device = torch.device('cuda')
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    
    # Enable multi-GPU support for T4 x2
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        # For distributed training, we'll use device_map="auto" when loading the model
        print("Multi-GPU training enabled")
        
        # Additional memory management for multi-GPU setup
        torch.cuda.empty_cache()
        # Set memory allocation strategy to reduce fragmentation
        if hasattr(torch.cuda, 'memory_stats'):
            print("Initial GPU memory allocated:", torch.cuda.memory_allocated(0) / (1024**3), "GB")
else:
    device = torch.device('cpu')
    print("Using CPU - Note: Training will be much slower on CPU")

# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Dataset configuration - using the same dataset as the original notebook
DATASET_ID = "mvasiliniuc/iva-swift-codeint"

# Model configuration - using Phi-3-mini-128k-instruct
MODEL_NAME = "microsoft/Phi-3-mini-128k-instruct"
MAX_LENGTH = 4096  # Phi-3 can handle long sequences natively
BATCH_SIZE = 2  # Reduced batch size for multi-GPU training (each GPU will process this batch size)
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
NUM_EPOCHS = 3
WARMUP_RATIO = 0.03
GRADIENT_ACCUMULATION_STEPS = 4  # Reduced since we're using 2 GPUs

# LoRA configuration
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

# Debug mode for testing with smaller dataset
DEBUG_MODE = False
DEBUG_SAMPLE_SIZE = 100

print(f"Using model: {MODEL_NAME}")
print(f"Max sequence length: {MAX_LENGTH}")
print(f"Batch size: {BATCH_SIZE} per device")
print(f"Effective batch size: {BATCH_SIZE * (2 if torch.cuda.device_count() > 1 else 1) * GRADIENT_ACCUMULATION_STEPS}")
print(f"LoRA rank: {LORA_R}")

# Function to load dataset with retry logic
def load_dataset_with_retry(dataset_id, max_retries=3, retry_delay=5):
    """Load a dataset with retry logic."""
    for attempt in range(max_retries):
        try:
            print(f"Loading dataset (attempt {attempt+1}/{max_retries})...")
            data = load_dataset(dataset_id, trust_remote_code=True)
            print(f"Dataset loaded successfully with {len(data['train'])} examples")
            return data
        except Exception as e:
            print(f"Error loading dataset (attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print("Maximum retries reached. Could not load dataset.")
                raise

# Load the dataset with retry logic
try:
    print(f"Loading dataset: {DATASET_ID}")
    data = load_dataset_with_retry(DATASET_ID)
    print("Dataset structure:")
    print(data)
    
    # If in debug mode, take a small sample of the dataset
    if DEBUG_MODE and 'train' in data:
        print(f"DEBUG MODE: Sampling {DEBUG_SAMPLE_SIZE} examples from dataset")
        # Take a stratified sample if possible
        data['train'] = data['train'].shuffle(seed=42).select(range(min(DEBUG_SAMPLE_SIZE, len(data['train']))))
        print(f"Reduced dataset size: {len(data['train'])} examples")
        
except Exception as e:
    print(f"Fatal error loading dataset: {e}")
    raise

# Verify dataset structure and column names
def verify_dataset_structure(dataset):
    """Verify that the dataset has the expected structure and columns."""
    required_columns = ['repo_name', 'path', 'content']
    if 'train' not in dataset:
        print("WARNING: Dataset does not have a 'train' split.")
        return False
    
    missing_columns = [col for col in required_columns if col not in dataset['train'].column_names]
    if missing_columns:
        print(f"WARNING: Dataset is missing required columns: {missing_columns}")
        return False
    
    print("Dataset structure verification passed.")
    return True

# Verify dataset structure
dataset_valid = verify_dataset_structure(data)
if not dataset_valid:
    print("Dataset structure is not as expected. Proceeding with caution.")

# Load the Phi-3 tokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, model_max_length=MAX_LENGTH)
    # Add padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"Tokenizer vocabulary size: {len(tokenizer)}")
    print(f"Tokenizer type: {tokenizer.__class__.__name__}")
except Exception as e:
    print(f"Error loading tokenizer: {e}")
    raise

def extract_file_type(path):
    """
    Extract the file type/category based on the file path and naming conventions in Swift projects.
    
    Args:
        path (str): The file path
        
    Returns:
        int: The category label (0-5)
    """
    path_lower = path.lower()
    filename = path.split('/')[-1].lower()
    
    # Category 0: Models - Data structures and model definitions
    if ('model' in path_lower or 
        'struct' in path_lower or 
        'entity' in path_lower or
        'data' in path_lower and 'class' in path_lower):
        return 0
    
    # Category 1: Views - UI related files
    elif ('view' in path_lower or 
          'ui' in path_lower or 
          'screen' in path_lower or 
          'page' in path_lower or
          'controller' in path_lower and 'view' in path_lower):
        return 1
    
    # Category 2: Controllers - Application logic
    elif ('controller' in path_lower or 
          'manager' in path_lower or 
          'coordinator' in path_lower or
          'service' in path_lower):
        return 2
    
    # Category 3: Utilities - Helper functions and extensions
    elif ('util' in path_lower or 
          'helper' in path_lower or 
          'extension' in path_lower or
          'common' in path_lower):
        return 3
    
    # Category 4: Tests - Test files
    elif ('test' in path_lower or 
          'spec' in path_lower or 
          'mock' in path_lower):
        return 4
    
    # Category 5: Configuration - Package and configuration files
    elif ('package.swift' in path_lower or 
          'config' in path_lower or 
          'settings' in path_lower or
          'info.plist' in path_lower):
        return 5
    
    # Default to category 3 (Utilities) if no clear category is found
    return 3

# Define category names for better readability
category_names = {
    0: "Models",
    1: "Views",
    2: "Controllers",
    3: "Utilities",
    4: "Tests",
    5: "Configuration"
}

# Apply the function to create labels
try:
    # Create a new column with the extracted labels
    labeled_data = data['train'].map(lambda example: {
        **example,
        'label': extract_file_type(example['path'])
    })
    
    # Count the distribution of labels
    label_counts = collections.Counter(labeled_data['label'])
    
    print("Label distribution:")
    for label, count in sorted(label_counts.items()):
        category_name = category_names.get(label, f"Unknown-{label}")
        print(f"Label {label} ({category_name}): {count} examples ({count/len(labeled_data)*100:.2f}%)")
    
    # Get unique labels
    unique_labels = sorted(label_counts.keys())
    num_labels = len(unique_labels)
    
    print(f"\nTotal unique labels: {num_labels}")
except Exception as e:
    print(f"Error in data preparation: {e}")
    raise

# Split the data into train, validation, and test sets
try:
    # Shuffle the data
    shuffled_data = labeled_data.shuffle(seed=42)
    
    # Split into train (80%), validation (10%), and test (10%)
    train_size = int(0.8 * len(shuffled_data))
    val_size = int(0.1 * len(shuffled_data))
    
    train_data = shuffled_data.select(range(train_size))
    val_data = shuffled_data.select(range(train_size, train_size + val_size))
    test_data = shuffled_data.select(range(train_size + val_size, len(shuffled_data)))
    
    print(f"Training set size: {len(train_data)}")
    print(f"Training set label distribution: {collections.Counter(train_data['label'])}")
    print(f"Validation set size: {len(val_data)}")
    print(f"Validation set label distribution: {collections.Counter(val_data['label'])}")
    print(f"Test set size: {len(test_data)}")
    print(f"Test set label distribution: {collections.Counter(test_data['label'])}")
except Exception as e:
    print(f"Error splitting data: {e}")
    raise

# Create instruction-based prompts for the model
def create_instruction_prompt(example):
    """Convert a code example into an instruction-based prompt for language learning."""
    code = example['content']
    label = example['label']
    category = category_names.get(label, f"Unknown-{label}")
    
    # Create different types of prompts# Check the created file
cat phi-train-modified.py | wc -l
# Let's make sure the AQLM part is correctly incorporated
grep -A 15 "Configure 2-bit AQLM quantization" phi-train-modified.py
# Now convert the Python script back to a notebook using jupytext
jupytext --to notebook phi-train-modified.py -o phi-train-aqlm.ipynb
# Check if the notebook was created
ls -la phi-train-aqlm.ipynb
# Seems like the cat output is too long, let's check if the important parts were saved
cat phi-train-modified.py | tail -n 50 | head -n 10
# Verify if the file was created completely and check its size
du -h phi-train-modified.py
# Let's try the jupytext conversion again
jupytext --to notebook phi-train-modified.py -o phi-train-aqlm.ipynb
ls -la phi-train-aqlm.ipynb
# Check the file size and see if it was created properly
ls -l phi-train-modified.py

# Let's try to create the file in smaller chunks
head -n 100 phi-train-modified.py > phi-train-aqlm.py
cat phi-train-modified.py | grep -n "aqlm_config ="

# Let's just create a smaller version with the key changes
cat > phi-train-aqlm-minimal.py << 'EOL'
#!/usr/bin/env python
# coding: utf-8

# # Training Phi-3-mini-128k-instruct with AQLM 2-bit Quantization
# 
# This notebook trains Microsoft's Phi-3-mini-128k-instruct model with AQLM 2-bit quantization to understand and work with Swift code.

# Install required libraries
# !pip install transformers datasets evaluate torch scikit-learn tqdm dropbox requests accelerate peft aqlm

import os
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from aqlm import AqlmConfig

# Model configuration - using Phi-3-mini-128k-instruct
MODEL_NAME = "microsoft/Phi-3-mini-128k-instruct"
MAX_LENGTH = 4096  # Phi-3 can handle long sequences natively
BATCH_SIZE = 2  # Reduced batch size for multi-GPU training

# LoRA configuration
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

# Configure 2-bit AQLM quantization (replacing the previous 4-bit BitsAndBytes)
print("Setting up 2-bit AQLM quantization...")
aqlm_config = AqlmConfig(
    bits=2,                        # Use 2-bit quantization
    device_map="auto",             # Automatically distribute model across available GPUs
    max_memory=None,               # Use maximum available memory
    offload_folder="aqlm_offload", # Folder for offloading to disk if needed
    trust_remote_code=True,        # Trust remote code for model loading
    dtype="float16"                # Use float16 for remaining parameters
)

# Load model with AQLM 2-bit quantization
try:
    print(f"Loading {MODEL_NAME} with AQLM 2-bit quantization...")
    
    # Load the model with AQLM 2-bit quantization
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=aqlm_config,
        device_map="auto",
        torch_dtype=torch.float16,
        use_cache=False
    )
    
    # Prepare the model for training
    model = prepare_model_for_kbit_training(model)
    
    # Apply LoRA for parameter-efficient fine-tuning
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    
    model = get_peft_model(model, lora_config)
    
    print("Model loaded successfully with AQLM 2-bit quantization!")
except Exception as e:
    print(f"Error loading model: {e}")
    import traceback
    traceback.print_exc()
