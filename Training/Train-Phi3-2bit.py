# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Training Phi-3-mini-128k-instruct to Learn Swift Programming Language
#
# This notebook trains Microsoft's Phi-3-mini-128k-instruct model to understand and work with Swift code using a dataset of real Swift files.

# %%
# Install required libraries
!pip install transformers datasets evaluate torch scikit-learn tqdm dropbox requests accelerate peft bitsandbytes
# Set PyTorch memory management environment variables to avoid fragmentation
import os
# Critical memory management settings to avoid OOM errors
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # Explicitly set to use 2 GPUs
# Set a smaller per-process memory fraction to leave room for other processes
os.environ["CUDA_MEM_FRACTION"] = "0.85"  # Reserve 15% for system and other CUDA processes

# %%
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
    BitsAndBytesConfig
)
from transformers.trainer_callback import EarlyStoppingCallback
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
# Import AQLM for 2-bit quantization
try:
    try:
        import aqlm
    except ImportError:
        print("AQLM package not found. Installing...")
        import subprocess
        subprocess.check_call(["pip", "install", "-U", "aqlm", "--no-cache-dir"])
        import aqlm
    
    # AQLM correctly imported - we'll use it directly for quantization when loading the model
    print("AQLM imported successfully - version:", aqlm.__version__ if hasattr(aqlm, "__version__") else "unknown")
except Exception as e:
    print(f"Error importing AQLM: {e}")
    print("Will fallback to 4-bit quantization using BitsAndBytes")

# Define enhanced memory cleanup function
def cleanup_memory(aggressive=False):
    """
    Clean up GPU memory to avoid fragmentation.
    
    Args:
        aggressive (bool): If True, performs more aggressive cleanup operations
    """
    print("Cleaning up memory...")
    # Standard cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # Additional aggressive cleanup
        if aggressive:
            print("Performing aggressive memory cleanup...")
            # Force a second round of garbage collection
            for _ in range(3):
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # If possible, clear CUDA IPC caches which can leak memory
            if hasattr(torch.cuda, '_sleep'):
                torch.cuda._sleep(1000)  # Short sleep to allow background cleanup
            
            # Log memory status after cleanup
            print("Memory after aggressive cleanup:")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.memory_allocated(i) / (1024**3):.2f} GB allocated, "
                      f"{torch.cuda.memory_reserved(i) / (1024**3):.2f} GB reserved")
        
# Define enhanced resource monitoring function
def monitor_resources(show_memory_details=False):
    """
    Monitor system and GPU resources with detailed memory breakdown.
    
    Args:
        show_memory_details (bool): If True, shows detailed memory stats per GPU
    """
    # Monitor CPU and RAM
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    mem = psutil.virtual_memory()
    print(f"CPU memory usage: {memory_info.rss / 1024 / 1024:.2f} MB ({memory_info.rss / mem.total * 100:.1f}% of system RAM)")
    print(f"System RAM: {mem.used / 1024 / 1024 / 1024:.2f} GB used of {mem.total / 1024 / 1024 / 1024:.2f} GB total")
    
    # Monitor GPU if available
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"Number of GPUs: {num_gpus}")
        
        total_allocated = 0
        total_reserved = 0
        
        for i in range(num_gpus):
            if hasattr(torch.cuda, 'memory_allocated'):
                allocated_gb = torch.cuda.memory_allocated(i) / (1024**3)
                reserved_gb = torch.cuda.memory_reserved(i) / (1024**3)
                total_allocated += allocated_gb
                total_reserved += reserved_gb
                
                # Get total memory for this GPU
                device_props = torch.cuda.get_device_properties(i)
                total_memory = device_props.total_memory / (1024**3)
                free_memory = (device_props.total_memory - torch.cuda.memory_allocated(i) - 
                              (torch.cuda.memory_reserved(i) - torch.cuda.memory_allocated(i))) / (1024**3)
                
                print(f"GPU {i} ({torch.cuda.get_device_name(i)})")
                print(f"  Memory: {allocated_gb:.2f} GB allocated, {reserved_gb:.2f} GB reserved")
                print(f"  Free: {free_memory:.2f} GB / Total: {total_memory:.2f} GB ({free_memory/total_memory*100:.1f}% free)")
                
                # Show detailed memory stats if requested
                if show_memory_details and hasattr(torch.cuda, 'memory_stats'):
                    try:
                        stats = torch.cuda.memory_stats(i)
                        print(f"  Detailed memory statistics for GPU {i}:")
                        if 'active_bytes.all.current' in stats:
                            print(f"    Active memory: {stats['active_bytes.all.current'] / (1024**3):.2f} GB")
                        if 'inactive_split_bytes.all.current' in stats:
                            print(f"    Inactive splits: {stats['inactive_split_bytes.all.current'] / (1024**3):.2f} GB")
                        if 'allocated_bytes.all.current' in stats:
                            print(f"    Allocated: {stats['allocated_bytes.all.current'] / (1024**3):.2f} GB")
                        if 'reserved_bytes.all.current' in stats:
                            print(f"    Reserved: {stats['reserved_bytes.all.current'] / (1024**3):.2f} GB")
                        if 'active_bytes.all.peak' in stats:
                            print(f"    Peak active memory: {stats['active_bytes.all.peak'] / (1024**3):.2f} GB")
                        if 'segment_size' in stats and 'allocated_bytes.all.allocated' in stats:
                            fragmentation = 1.0 - (stats['allocated_bytes.all.allocated'] / stats['segment_size'])
                            print(f"    Fragmentation estimate: {fragmentation:.1%}")
                    except RuntimeError as e:
                        print(f"  Could not get detailed memory stats: {e}")
        
        print(f"Total GPU memory: {total_allocated:.2f} GB allocated, {total_reserved:.2f} GB reserved")


# %%
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

# %%
# Dataset configuration - using the same dataset as the original notebook
DATASET_ID = "mvasiliniuc/iva-swift-codeint"

# Model configuration - using Phi-3-mini-128k-instruct
MODEL_NAME = "microsoft/Phi-3-mini-128k-instruct"
MAX_LENGTH = 2048  # Reduced from 4096 to save memory
BATCH_SIZE = 1     # Reduced batch size per GPU to prevent OOM errors
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
NUM_EPOCHS = 3
WARMUP_RATIO = 0.03
GRADIENT_ACCUMULATION_STEPS = 8  # Increased to compensate for smaller batch size
# Effective batch size = BATCH_SIZE * num_gpus * GRADIENT_ACCUMULATION_STEPS = 1 * 2 * 8 = 16

# LoRA configuration - optimized for memory efficiency
LORA_R = 8         # Reduced from 16 to save memory
LORA_ALPHA = 16    # Reduced proportionally
LORA_DROPOUT = 0.05

# Memory optimization
USE_FP16 = True    # Use half precision for additional memory savings
USE_ACTIVATION_CHECKPOINTING = True  # Trade computation for memory
OPTIMIZE_CUDA_GRAPH = True  # Optimize CUDA execution when possible

# Multi-GPU configuration
DEVICE_MAP_STRATEGY = "balanced_low_0"  # Options: "auto", "balanced", "balanced_low_0"
USE_MEMORY_EFFICIENT_ATTENTION = True

# Debug mode for testing with smaller dataset
DEBUG_MODE = False
DEBUG_SAMPLE_SIZE = 100

print(f"Using model: {MODEL_NAME}")
print(f"Max sequence length: {MAX_LENGTH}")
print(f"Batch size: {BATCH_SIZE} per device")
print(f"Effective batch size: {BATCH_SIZE * (2 if torch.cuda.device_count() > 1 else 1) * GRADIENT_ACCUMULATION_STEPS}")
print(f"LoRA rank: {LORA_R}")


# %%
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


# %%
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

# %%
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


# %%
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

# %%
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

# %%
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


# %%
# Create instruction-based prompts for the model
def create_instruction_prompt(example):
    """Convert a code example into an instruction-based prompt for language learning."""
    code = example['content']
    label = example['label']
    category = category_names.get(label, f"Unknown-{label}")
    
    # Create different types of prompts to help the model learn the language
    prompt_types = [
        # Explain code functionality
        "Explain what this Swift code does and how it works:\n\n",
        
        # Identify patterns and features
        "Identify and explain the key Swift language features used in this code:\n\n",
        
        # Complete or extend code
        "Complete or extend this Swift code with appropriate functionality:\n\n",
        
        # Fix or improve code
        "Suggest improvements or best practices for this Swift code:\n\n",
        
        # Understand code structure
        f"This is a Swift {category.lower()} file. Explain its structure and purpose:\n\n",
        
        # Code generation tasks
        "Write a Swift function that accomplishes the same task as this code but more efficiently:\n\n",
        
        # Language understanding
        "Explain the Swift syntax and language features demonstrated in this code:\n\n",
        
        # Learning from examples
        "Study this Swift code example and explain what you can learn from it:\n\n"
    ]
    
    # Select a random prompt type
    instruction = random.choice(prompt_types)
    
    code_section = f"```swift\n{code}\n```\n\n"
    
    # Create the full prompt
    prompt = instruction + code_section
    
    # Create a detailed response based on the prompt type and code category
    if "Explain what this Swift code does" in instruction:
        response = f"This Swift code is a {category.lower()} file that "
        if category == "Models":
            response += "defines data structures and model objects. "
        elif category == "Views":
            response += "implements user interface components. "
        elif category == "Controllers":
            response += "manages application logic and coordinates between models and views. "
        elif category == "Utilities":
            response += "provides helper functions and extensions. "
        elif category == "Tests":
            response += "contains test cases to verify functionality. "
        elif category == "Configuration":
            response += "configures application settings and parameters. "
        
        response += "The code uses Swift syntax with "
        
        # Add some language-specific details based on code content
        if "class" in code:
            response += "class definitions, "
        if "struct" in code:
            response += "struct definitions, "
        if "func" in code:
            response += "function declarations, "
        if "var" in code:
            response += "variable declarations, "
        if "let" in code:
            response += "constant declarations, "
        if "guard" in code or "if let" in code:
            response += "optional unwrapping, "
        if "extension" in code:
            response += "extensions, "
        if "protocol" in code:
            response += "protocol implementations, "
            
        # Remove trailing comma and space if present
        if response.endswith(", "):
            response = response[:-2] + "."
        else:
            response += "various Swift features."
    
    elif "Identify and explain the key Swift language features" in instruction:
        response = "This Swift code demonstrates several key language features:\n\n"
        
        # Add language features based on code content
        features = []
        if "class" in code:
            features.append("1. **Classes**: Swift classes are reference types that support inheritance and reference counting.")
        if "struct" in code:
            features.append("1. **Structs**: Swift structs are value types that are copied when assigned or passed as arguments.")
        if "protocol" in code:
            features.append("1. **Protocols**: Similar to interfaces in other languages, protocols define a blueprint of methods, properties, and requirements.")
        if "extension" in code:
            features.append("1. **Extensions**: Swift allows adding functionality to existing types through extensions.")
        if "guard" in code:
            features.append("1. **Guard statements**: Used for early returns and unwrapping optionals, improving code readability.")
        if "if let" in code or "guard let" in code:
            features.append("1. **Optional binding**: Swift's way of safely unwrapping optional values.")
        if "enum" in code:
            features.append("1. **Enumerations**: Swift enums are first-class types that can have methods and computed properties.")
        if "func" in code:
            features.append("1. **Functions**: Swift functions can have parameters, return values, and support closures.")
        
        # If no specific features were identified, add a generic response
        if not features:
            features.append("1. **Swift syntax**: The code demonstrates standard Swift syntax and conventions.")
            features.append("2. **Type safety**: Swift's strong type system helps prevent errors at compile time.")
            features.append("3. **Readability**: Swift's clean syntax makes code easy to read and maintain.")
        
        # Renumber the features
        for i, feature in enumerate(features):
            feature_parts = feature.split(": ", 1)
            if len(feature_parts) == 2:
                features[i] = f"{i+1}. **{feature_parts[0].split('**')[1]}**: {feature_parts[1]}"
        
        response += "\n".join(features)
    
    elif "Complete or extend this Swift code" in instruction or "Write a Swift function" in instruction:
        # For code generation tasks, provide a thoughtful response about how to approach the task
        response = f"To extend this Swift {category.lower()} code, I would consider the following approach:\n\n"
        
        if category == "Models":
            response += "1. Add additional properties to capture more data attributes\n"
            response += "2. Implement Codable protocol for easy JSON serialization\n"
            response += "3. Add validation methods to ensure data integrity\n"
            response += "4. Include computed properties for derived values\n\n"
            response += "Here's an implementation example:\n\n```swift\n"
            
            if "struct" in code:
                response += "// Extension to add Codable conformance\nextension MyStruct: Codable {\n    // Codable implementation\n}\n\n"
                response += "// Add validation method\nextension MyStruct {\n    func validate() -> Bool {\n        // Validation logic\n        return true\n    }\n}\n"
            else:
                response += "// Example extension or additional functionality\n// that would be appropriate for this model\n"
            
            response += "```"
            
        elif category == "Views":
            response += "1. Add UI customization options\n"
            response += "2. Implement additional user interaction handlers\n"
            response += "3. Add accessibility support\n"
            response += "4. Implement view lifecycle methods\n\n"
            response += "Here's an implementation example:\n\n```swift\n"
            response += "// Example extension or additional functionality\n// that would be appropriate for this view\n"
            response += "```"
            
        else:
            response += "1. Add error handling to make the code more robust\n"
            response += "2. Implement additional helper methods\n"
            response += "3. Add documentation comments to improve code readability\n"
            response += "4. Consider performance optimizations where appropriate\n\n"
            response += "Here's an implementation example:\n\n```swift\n"
            response += "// Example extension or additional functionality\n// that would be appropriate for this code\n"
            response += "```"
    
    else:
        # Generic response for other prompt types
        response = f"This Swift code demonstrates typical patterns used in {category.lower()} files. "
        response += "It follows Swift language conventions and showcases proper syntax for defining "
        
        if category == "Models":
            response += "data structures with properties and methods. Swift models typically use structs for value semantics or classes when reference semantics are needed. The code demonstrates Swift's strong typing system and property declarations."
        elif category == "Views":
            response += "UI components with layout and interaction logic. Swift views often use UIKit or SwiftUI frameworks, with clear separation of UI elements and their behaviors. The code shows how Swift handles user interface components and event responses."
        elif category == "Controllers":
            response += "application logic and coordination between components. Controllers in Swift manage the flow of data between models and views, implementing business logic and handling user interactions. The code demonstrates Swift's approach to application architecture."
        elif category == "Utilities":
            response += "helper functions and extensions to enhance functionality. Swift utilities often leverage the language's powerful extension capabilities to add functionality to existing types. The code shows how Swift can be extended and customized through utility functions."
        elif category == "Tests":
            response += "test cases with setup, execution, and verification steps. Swift tests typically use XCTest framework with arrange-act-assert pattern. The code demonstrates Swift's approach to unit testing and verification."
        elif category == "Configuration":
            response += "application settings and configuration parameters. Swift configuration files often define constants, environment settings, and application parameters. The code shows how Swift handles application configuration and settings management."
    
    # Combine prompt and response for instruction tuning
    full_text = f"<|user|>\n{prompt}\n<|assistant|>\n{response}\n"
    
    return {
        "text": full_text,
        "prompt": prompt,
        "response": response,
        "label": label,
        "category": category
    }


# %%
# Apply the function to create instruction-based datasets
try:
    # Create instruction datasets
    train_instructions = train_data.map(create_instruction_prompt)
    val_instructions = val_data.map(create_instruction_prompt)
    test_instructions = test_data.map(create_instruction_prompt)
    
    # Print an example to verify
    print("Example instruction prompt:")
    print("-" * 80)
    print(train_instructions[0]['text'])
    print("-" * 80)
    
    print(f"Created {len(train_instructions)} training instructions")
    print(f"Created {len(val_instructions)} validation instructions")
    print(f"Created {len(test_instructions)} test instructions")
except Exception as e:
    print(f"Error creating instruction prompts: {e}")
    raise


# %%
# FIXED: Tokenize the instruction data with proper handling of padding and truncation
def tokenize_instruction(examples):
    """Tokenize the instruction text with explicit padding and truncation settings."""
    # Process one example at a time to avoid dimension issues
    results = {"input_ids": [], "attention_mask": [], "labels": []}
    
    for text in examples['text']:
        # Tokenize with explicit padding and truncation settings
        encoded = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors=None  # Return Python lists, not PyTorch tensors
        )
        
        # Add to results
        results["input_ids"].append(encoded["input_ids"])
        results["attention_mask"].append(encoded["attention_mask"])
        results["labels"].append(encoded["input_ids"].copy())  # Copy input_ids for labels
    
    return results


# %%
try:
    # Apply tokenization to each split
    tokenized_train = train_instructions.map(
        tokenize_instruction,
        batched=True,
        remove_columns=['repo_name', 'path', 'content', 'label', 'text', 'prompt', 'response', 'category']
    )
    
    tokenized_val = val_instructions.map(
        tokenize_instruction,
        batched=True,
        remove_columns=['repo_name', 'path', 'content', 'label', 'text', 'prompt', 'response', 'category']
    )
    
    # Set the format for PyTorch
    tokenized_train.set_format("torch")
    tokenized_val.set_format("torch")
    
    print(f"Tokenized {len(tokenized_train)} training examples")
    print(f"Tokenized {len(tokenized_val)} validation examples")
    print("Data tokenization complete")
except Exception as e:
    print(f"Error tokenizing data: {e}")
    raise

# %%
# Set up training arguments with aggressive memory optimization for multi-GPU training
try:
    # Create output directory if it doesn't exist
    os.makedirs("./phi3_swift_model", exist_ok=True)
    
    # Create logs directory
    os.makedirs("./logs", exist_ok=True)
    
    # Determine optimal dataloader workers - too many can consume memory
    if torch.cuda.is_available():
        # Use 2 workers per GPU - balance between speed and memory usage
        num_workers = min(2, os.cpu_count() // 2)
    else:
        num_workers = 0  # No workers on CPU-only setup
    
    print(f"Using {num_workers} dataloader workers")
    
    # Configure training arguments with memory-optimized settings
    training_args = TrainingArguments(
        output_dir="./phi3_swift_model",
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        logging_dir="./logs",
        logging_steps=10,
        # Save model less frequently to reduce I/O overhead
        save_steps=1000,
        save_total_limit=1,  # Keep only the best model
        # Evaluate less frequently to save memory
        eval_strategy="steps",
        eval_steps=1000,
        load_best_model_at_end=True,
        # Memory optimization settings
        fp16=USE_FP16,  # Use half precision
        bf16=False,     # Don't use bfloat16 (not as widely supported on older GPUs)
        gradient_checkpointing=USE_ACTIVATION_CHECKPOINTING,  # Enable gradient checkpointing
        optim="adamw_torch_fused",  # Use fused optimizer if available for better memory efficiency
        # Avoid memory fragmentation during backward pass
        max_grad_norm=1.0,
        # Distributed training parameters
        local_rank=int(os.environ.get("LOCAL_RANK", -1)),
        ddp_find_unused_parameters=False,  # Optimize DDP
        ddp_bucket_cap_mb=50,  # Smaller communication buckets to avoid OOM
        dataloader_num_workers=num_workers,
        dataloader_pin_memory=True,        # Pin memory for faster data transfer
        report_to="none",                  # Disable reporting to avoid extra overhead
        # Reduce peak memory usage with sequential model initialization
        ddp_timeout=7200,                  # Long timeout for slow startup
        # Avoid memory leaks with explicit no_cuda_filter
        use_legacy_prediction_loop=False,  # Modern prediction loop is more memory efficient
        # Avoid redundant operations
        label_smoothing_factor=0.0,        # Disable label smoothing to save computation
        disable_tqdm=False,                # Keep progress bars for monitoring
        # Remove unnecessary memory usage
        remove_unused_columns=True,        # Remove columns not used by the model
        # Avoid memory fragmentation with larger seed
        seed=42 + torch.cuda.device_count() if torch.cuda.is_available() else 42,
    )
    
    # Check if we need to add DeepSpeed config for even more memory efficiency
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print("Multi-GPU detected, adding additional memory optimizations")
        # Set additional attributes for memory optimization in multi-GPU setup
        training_args.gradient_accumulation_mode = "steps"  # More memory efficient accumulation
        
        # Try to enable DeepSpeed Zero-2 if more than 2 GPUs
        if torch.cuda.device_count() > 2 and not DEBUG_MODE:
            print("Enabling DeepSpeed ZeRO-2 for advanced GPU memory optimization")
            ds_config = {
                "zero_optimization": {
                    "stage": 2,
                    "offload_optimizer": {
                        "device": "cpu",
                        "pin_memory": True
                    },
                    "contiguous_gradients": True,
                    "overlap_comm": True,
                    "reduce_scatter": True,
                    "reduce_bucket_size": 5e7,
                    "allgather_bucket_size": 5e7
                },
                "fp16": {
                    "enabled": USE_FP16,
                    "loss_scale": 0,
                    "loss_scale_window": 1000,
                    "initial_scale_power": 16,
                    "hysteresis": 2,
                    "min_loss_scale": 1
                }
            }
            training_args.deepspeed = ds_config
    
    print(f"Training arguments configured for {'multi-GPU' if torch.cuda.device_count() > 1 else 'single-GPU'} training")
    print(f"Using gradient checkpointing: {training_args.gradient_checkpointing}")
    print(f"Using mixed precision: {training_args.fp16}")
    print(f"Gradient accumulation steps: {training_args.gradient_accumulation_steps}")
    print(f"Local rank: {training_args.local_rank}")
    
    # Clean up memory after setting up training args
    cleanup_memory()
    
except Exception as e:
    print(f"Error setting up training arguments: {e}")
    traceback.print_exc()
    raise

# %%
# Define early stopping callback
early_stopping_callback = EarlyStoppingCallback(
    early_stopping_patience=3,
    early_stopping_threshold=0.01
)


# %%
# Create a memory-efficient custom data collator
class MemoryEfficientDataCollator(DataCollatorForLanguageModeling):
    """
    Data collator optimized for memory efficiency with multi-GPU training.
    Includes safeguards against OOM errors and proper batch construction.
    """
    def __init__(self, tokenizer, mlm=False, max_length=None):
        super().__init__(tokenizer=tokenizer, mlm=mlm)
        self.max_length = max_length
        self.pad_token_id = tokenizer.pad_token_id
        self.device = "cpu"  # Keep batches on CPU until explicitly moved to minimize GPU memory

    def __call__(self, features):
        # Safeguard: check if any feature is None or missing required fields
        if not features or any(f is None for f in features):
            print("Warning: Empty or None features received by data collator")
            # Return a minimal valid batch to avoid crashes
            if torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
            return {
                "input_ids": torch.zeros((1, 1), dtype=torch.long, device=device),
                "attention_mask": torch.zeros((1, 1), dtype=torch.long, device=device),
                "labels": torch.zeros((1, 1), dtype=torch.long, device=device),
            }
        
        # Ensure all features have the same keys
        required_keys = ["input_ids", "attention_mask", "labels"]
        if not all(all(k in f for k in required_keys) for f in features):
            missing_keys = []
            for i, f in enumerate(features):
                if not all(k in f for k in required_keys):
                    missing = [k for k in required_keys if k not in f]
                    missing_keys.append(f"Feature {i} missing keys: {missing}")
            raise ValueError(f"Some features are missing required keys: {missing_keys[:5]}")
        
        # Safeguard: ensure all features have the expected tensor type
        for i, f in enumerate(features):
            for k in required_keys:
                if not isinstance(f[k], torch.Tensor):
                    features[i][k] = torch.tensor(f[k], dtype=torch.long, device=self.device)

        # Create batches efficiently: keep on CPU, use torch.cat instead of stack when possible
        try:
            # Concatenate for more memory efficiency
            batch = {
                "input_ids": torch.stack([f["input_ids"] for f in features]).to(self.device),
                "attention_mask": torch.stack([f["attention_mask"] for f in features]).to(self.device),
                "labels": torch.stack([f["labels"] for f in features]).to(self.device)
            }
            
            # Perform memory cleanup
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            return batch
        except Exception as e:
            print(f"Error creating batch: {e}")
            # Fall back to a more careful approach
            try:
                # Try with a more careful approach, one tensor at a time
                input_ids = torch.stack([f["input_ids"] for f in features]).to(self.device)
                attention_mask = torch.stack([f["attention_mask"] for f in features]).to(self.device)
                labels = torch.stack([f["labels"] for f in features]).to(self.device)
                
                return {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels
                }
            except RuntimeError as re:
                print(f"Critical error in batch creation: {re}")
                # As a last resort, return a minimal batch to avoid training failure
                print("Creating emergency minimal batch to avoid crash")
                return {
                    "input_ids": torch.ones((1, 1), dtype=torch.long, device=self.device),
                    "attention_mask": torch.ones((1, 1), dtype=torch.long, device=self.device),
                    "labels": torch.ones((1, 1), dtype=torch.long, device=self.device),
                }

# Create memory-efficient data collator for language modeling
print("Creating memory-efficient data collator...")
data_collator = MemoryEfficientDataCollator(
    tokenizer=tokenizer,
    mlm=False,  # We're doing causal language modeling, not masked language modeling
    max_length=MAX_LENGTH
)

# Force a small garbage collection to clean up any temporary tensors
gc.collect()

# %%
# Create a flag to track which quantization method we're using
USING_AQLM = False
QUANT_BITS = 2  # Default to 2-bit quantization

print(f"Loading {MODEL_NAME} with optimized multi-GPU configuration...")

# Configure multi-GPU device mapping
if torch.cuda.is_available() and torch.cuda.device_count() > 1:
    print(f"Configuring for {torch.cuda.device_count()} GPUs using strategy: {DEVICE_MAP_STRATEGY}")
    
    # Start with an empty cache to maximize available memory
    cleanup_memory(aggressive=True)
    
    if DEVICE_MAP_STRATEGY == "auto":
        device_map = "auto"
    elif DEVICE_MAP_STRATEGY == "balanced":
        # Manually distribute layers across GPUs for more balanced memory usage
        device_map = "balanced"
    elif DEVICE_MAP_STRATEGY == "balanced_low_0":
        # Custom balanced strategy that puts less memory on GPU 0
        # This helps if GPU 0 is sometimes used for other operations
        try:
            from accelerate import infer_auto_device_map
            from transformers.dynamic_module_utils import get_class_from_dynamic_module
            from transformers.modeling_utils import PreTrainedModel
            
            # Get the model class to determine number of layers
            config = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)
            model_class = get_class_from_dynamic_module(config.auto_map["AutoModelForCausalLM"], 
                                                       MODEL_NAME)
            dummy_model = model_class(config)
            
            # Get number of transformer layers
            if hasattr(config, "num_hidden_layers"):
                num_layers = config.num_hidden_layers
            elif hasattr(config, "n_layer"):
                num_layers = config.n_layer
            else:
                print("Could not determine number of layers, falling back to 'balanced'")
                device_map = "balanced"
                num_layers = 0
            
            if num_layers > 0:
                # Distribute more layers to GPU 1 to leave GPU 0 with more free memory
                device_map = {
                    "model.embed_tokens": 0,
                    "model.norm": 1,
                    "lm_head": 1
                }
                
                # Allocate first 1/3 of layers to GPU 0, remaining 2/3 to GPU 1
                gpu0_layers = num_layers // 3
                for i in range(num_layers):
                    if i < gpu0_layers:
                        device_map[f"model.layers.{i}"] = 0
                    else:
                        device_map[f"model.layers.{i}"] = 1
                        
                print(f"Custom device map created with {gpu0_layers} layers on GPU 0 and {num_layers - gpu0_layers} layers on GPU 1")
            
        except Exception as e:
            print(f"Error creating custom device map: {e}, falling back to 'balanced'")
            device_map = "balanced"
    else:
        # Default to auto if unknown strategy
        device_map = "auto"
        
    print(f"Using device_map: {device_map}")
else:
    device_map = None
    if torch.cuda.is_available():
        print("Only one GPU available, using it for all computations")
    else:
        print("No GPUs available, using CPU (training will be very slow)")

# Set floating point precision
torch_dtype = torch.float16 if USE_FP16 else torch.float32

# Configure memory-efficient attention if supported
attn_implementation = "flash_attention_2" if USE_MEMORY_EFFICIENT_ATTENTION else "eager"
print(f"Using attention implementation: {attn_implementation}")

try:
    # First check if AQLM is available for 2-bit quantization
    if 'aqlm' in globals() or 'aqlm' in locals():
        # Use AQLM's approach for 2-bit quantization
        print(f"Using AQLM for 2-bit quantization...")
        
        # First load the model normally with memory optimizations
        print(f"Loading base model {MODEL_NAME}...")
        
        # Calculate max memory per GPU to avoid OOM
        max_memory = {}
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                total_mem = torch.cuda.get_device_properties(i).total_memory
                # Reserve 90% of memory for model, leaving 10% for overhead
                max_memory[i] = f"{int(total_mem * 0.9 / 1024 / 1024)}MiB"
            print(f"Max memory per GPU: {max_memory}")
        
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True,
            use_cache=False,  # Disable KV cache during training
            low_cpu_mem_usage=True,
            max_memory=max_memory,
            attn_implementation=attn_implementation,
        )
        
        # Apply activation checkpointing if enabled
        if USE_ACTIVATION_CHECKPOINTING and hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
            print("Gradient checkpointing enabled")
        
        # Now apply AQLM 2-bit quantization to the model
        print("Applying AQLM 2-bit quantization...")
        
        try:
            # Find the AQLM quantize function
            if hasattr(aqlm, 'quantize'):
                quantize_fn = aqlm.quantize
            elif hasattr(aqlm, 'quantization') and hasattr(aqlm.quantization, 'quantize'):
                quantize_fn = aqlm.quantization.quantize
            else:
                # Try to discover the correct module
                for module_name in dir(aqlm):
                    module = getattr(aqlm, module_name)
                    if hasattr(module, 'quantize'):
                        quantize_fn = module.quantize
                        print(f"Found quantize function in aqlm.{module_name}")
                        break
                else:
                    raise ImportError("Could not find quantize function in AQLM modules")
            
            # Clean up memory before quantization
            cleanup_memory(aggressive=True)
            
            # Apply quantization with reduced LoRA rank for memory efficiency
            model = quantize_fn(
                model, 
                bits=2,  # Always try 2-bit first
                lora_rank=LORA_R,
            )
            USING_AQLM = True
            QUANT_BITS = 2
            print("Successfully applied AQLM 2-bit quantization")
            
            # Monitor memory after quantization
            print("Memory usage after quantization:")
            monitor_resources(show_memory_details=True)
            
        except Exception as quant_error:
            print(f"AQLM 2-bit quantization failed: {quant_error}")
            print("Trying AQLM with 4-bit quantization instead...")
            
            # Try with 4-bit AQLM quantization
            try:
                # Clean up memory before retry
                del model
                cleanup_memory(aggressive=True)
                
                # Reload the model
                model = AutoModelForCausalLM.from_pretrained(
                    MODEL_NAME,
                    torch_dtype=torch_dtype,
                    device_map=device_map,
                    trust_remote_code=True,
                    use_cache=False,
                    low_cpu_mem_usage=True,
                    max_memory=max_memory,
                    attn_implementation=attn_implementation,
                )
                
                # Apply activation checkpointing if enabled
                if USE_ACTIVATION_CHECKPOINTING and hasattr(model, "gradient_checkpointing_enable"):
                    model.gradient_checkpointing_enable()
                
                # Try 4-bit quantization with AQLM
                from aqlm import quantize
                model = quantize(
                    model,
                    bits=4,
                    lora_rank=LORA_R
                )
                USING_AQLM = True
                QUANT_BITS = 4
                print("Successfully applied AQLM 4-bit quantization")
            except Exception as e:
                print(f"AQLM 4-bit quantization also failed: {e}")
                raise  # Let it fall through to BitsAndBytes fallback
    else:
        raise ImportError("AQLM not available")
        
except Exception as e:
    # Fallback to using BitsAndBytes for 4-bit quantization
    print(f"Falling back to BitsAndBytes 4-bit quantization: {e}")
    QUANT_BITS = 4
    USING_AQLM = False
    
    # Clean up memory before loading with BitsAndBytes
    cleanup_memory(aggressive=True)
    
    # Calculate max memory per GPU to avoid OOM
    max_memory = {}
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            total_mem = torch.cuda.get_device_properties(i).total_memory
            # Reserve 90% of memory for model, leaving 10% for overhead
            max_memory[i] = f"{int(total_mem * 0.9 / 1024 / 1024)}MiB"
        print(f"Max memory per GPU: {max_memory}")
    
    # Configure BitsAndBytes for memory-efficient 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch_dtype,
        bnb_4bit_use_double_quant=True,   # Use nested quantization for more memory savings
        bnb_4bit_quant_storage=torch_dtype # Match storage type with compute type
    )
    
    # Load model with BitsAndBytes 4-bit quantization and memory optimizations
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map=device_map,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        use_cache=False,
        max_memory=max_memory,
        attn_implementation=attn_implementation,
    )
    
    # Apply activation checkpointing if enabled and supported
    if USE_ACTIVATION_CHECKPOINTING and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled")
        
    print("Successfully loaded model with BitsAndBytes 4-bit quantization")
    
    # Monitor memory after loading
    print("Memory usage after loading:")
    monitor_resources(show_memory_details=True)
    print("Successfully loaded model with BitsAndBytes 4-bit quantization")

# Configure LoRA for fine-tuning with optimized memory usage
print("Setting up LoRA fine-tuning with memory optimizations...")

# Get model architecture details to determine target modules
target_modules = None  # Will be automatically detected if None
model_prefix = ""

# Try to identify model architecture to set optimal target modules
try:
    model_architecture = model.config.architectures[0] if hasattr(model.config, "architectures") else "unknown"
    print(f"Detected model architecture: {model_architecture}")
    
    # Target specific modules based on model architecture
    if "Llama" in model_architecture:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        model_prefix = "model.layers."
    elif "Phi" in model_architecture or "phi" in model_architecture.lower():
        # Phi models use different module names
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        model_prefix = "model.layers."
    elif "GPT" in model_architecture or "Mistral" in model_architecture:
        target_modules = ["c_attn", "c_proj", "c_fc"]
        model_prefix = "transformer.h."
    else:
        # Fallback to auto-detection
        print("Using automatic target module detection")
        
    if target_modules:
        print(f"Using target modules for {model_architecture}: {target_modules}")
except Exception as e:
    print(f"Error detecting model architecture: {e}, will use default target modules")

# Configure memory-efficient LoRA
lora_config = LoraConfig(
    r=LORA_R,                           # Rank dimension
    lora_alpha=LORA_ALPHA,              # Alpha parameter for LoRA scaling
    lora_dropout=LORA_DROPOUT,          # Dropout probability for LoRA layers
    bias="none",                        # Don't train bias parameters to save memory
    task_type="CAUSAL_LM",              # Task type for causal language modeling
    target_modules=target_modules,      # Target specific modules for efficiency
    modules_to_save=None,               # Don't save any modules fully (use LoRA for everything)
    fan_in_fan_out=False,               # Set to True for models like GPT2
    inference_mode=False,               # Training mode
)

# Memory cleanup before preparing model
cleanup_memory(aggressive=True)

# Prepare the model for training with LoRA with additional safeguards
try:
    # First just prepare the model for k-bit training
    print("Preparing model for k-bit training...")
    model = prepare_model_for_kbit_training(model)
    
    # Another memory cleanup after preparation
    cleanup_memory()
    
    # Monitor memory before applying LoRA
    print("Memory before applying LoRA:")
    monitor_resources()
    
    # Now apply the LoRA configuration
    print("Applying LoRA configuration...")
    model = get_peft_model(model, lora_config)
    
    print(f"LoRA configuration applied successfully with rank={LORA_R}")
    # Count and report trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params:.2%} of total)")
except Exception as e:
    print(f"Error preparing model with LoRA: {e}")
    print("Attempting to recover with simpler LoRA configuration...")
    
    # Try again with simpler configuration if first attempt failed
    try:
        # Clean up from previous attempt
        cleanup_memory(aggressive=True)
        
        # Load model again if needed
        if 'model' not in locals() or model is None:
            print("Reloading model...")
            # Here we would reload the model, but since it should still be in memory, we'll skip
            # and just retry with a simpler LoRA config
        
        # Simpler LoRA config with fewer target modules
        lora_config = LoraConfig(
            r=4,  # Use smaller rank
            lora_alpha=8,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            # Target fewer modules
            target_modules=["q_proj", "v_proj"] if target_modules else None
        )
        
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, lora_config)
        print("Applied simplified LoRA configuration successfully")
    except Exception as recovery_error:
        print(f"Recovery also failed: {recovery_error}")
        raise  # If recovery also fails, we need to stop

# Monitor memory after LoRA setup
print("Memory after LoRA setup:")
monitor_resources(show_memory_details=True)

# Print information about the quantized model
quant_method = "AQLM" if USING_AQLM else "BitsAndBytes"
print(f"Model loaded and configured with {QUANT_BITS}-bit {quant_method} quantization and LoRA (rank={LORA_R})")
print(f"Model architecture: {model.__class__.__name__}")
print(f"Total parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M")

# %%
# Add memory monitoring callback to track GPU usage during training
class MemoryMonitorCallback(TrainerCallback):
    """Callback to monitor memory usage during training and prevent OOM errors."""
    
    def __init__(self, memory_threshold=0.95, check_interval=10):
        self.memory_threshold = memory_threshold  # Stop if GPU memory usage exceeds this fraction
        self.check_interval = check_interval      # Check every N steps
        self.last_check = 0
        self.peak_memory = 0.0
        self.had_memory_warning = False
    
    def on_step_end(self, args, state, control, **kwargs):
        # Only check every check_interval steps to reduce overhead
        if state.global_step - self.last_check < self.check_interval:
            return
        
        self.last_check = state.global_step
        
        try:
            if torch.cuda.is_available():
                # Check memory usage on all GPUs
                for i in range(torch.cuda.device_count()):
                    allocated = torch.cuda.memory_allocated(i) / torch.cuda.get_device_properties(i).total_memory
                    self.peak_memory = max(self.peak_memory, allocated)
                    
                    # If memory exceeds threshold, emit a warning
                    if allocated > self.memory_threshold and not self.had_memory_warning:
                        print(f"\n⚠️ WARNING: GPU {i} memory usage is {allocated:.1%}, above threshold {self.memory_threshold:.1%}")
                        print("Attempting to free memory...")
                        
                        # Try to free some memory
                        torch.cuda.empty_cache()
                        gc.collect()
                        
                        # Record that we've had a warning so we don't spam
                        self.had_memory_warning = True
                        
                        # If memory is critically high, try more aggressive recovery
                        if allocated > 0.98:
                            print("CRITICAL: Memory nearly exhausted. Taking emergency measures...")
                            try:
                                # Log peak memory to help debugging
                                if hasattr(torch.cuda, 'memory_stats'):
                                    stats = torch.cuda.memory_stats(i)
                                    if 'active_bytes.all.peak' in stats:
                                        peak_bytes = stats['active_bytes.all.peak']
                                        print(f"Peak memory usage: {peak_bytes / 1024**3:.2f} GB")
                            except:
                                pass
        except Exception as e:
            print(f"Error in memory monitoring: {e}")
    
    def on_epoch_end(self, args, state, control, **kwargs):
        if torch.cuda.is_available():
            print(f"\nEpoch {state.epoch} completed. Peak memory usage: {self.peak_memory:.1%}.")
            # Reset the warning flag for the next epoch
            self.had_memory_warning = False
            # Perform memory cleanup between epochs
            torch.cuda.empty_cache()
            gc.collect()

# Add safe training wrapper
def safe_training_loop(trainer, max_retries=2):
    """Run training with safeguards against OOM errors."""
    for retry in range(max_retries + 1):
        try:
            return trainer.train()
        except RuntimeError as e:
            if "CUDA out of memory" in str(e) and retry < max_retries:
                print(f"\n\nCUDA OOM error detected (attempt {retry+1}/{max_retries+1}). Attempting recovery...")
                # Aggressive memory cleanup
                torch.cuda.empty_cache()
                gc.collect()
                time.sleep(5)  # Give system time to stabilize
                
                # Check if we can reduce batch size for recovery
                if hasattr(trainer.args, 'per_device_train_batch_size') and trainer.args.per_device_train_batch_size > 1:
                    old_batch_size = trainer.args.per_device_train_batch_size
                    trainer.args.per_device_train_batch_size = max(1, old_batch_size // 2)
                    print(f"Reducing batch size from {old_batch_size} to {trainer.args.per_device_train_batch_size}")
                    
                    # Adjust gradient accumulation to maintain same effective batch size
                    old_grad_accum = trainer.args.gradient_accumulation_steps
                    trainer.args.gradient_accumulation_steps = old_grad_accum * 2
                    print(f"Increasing gradient accumulation from {old_grad_accum} to {trainer.args.gradient_accumulation_steps}")
                else:
                    print("Cannot reduce batch size further, already at minimum")
                    
                print("Retrying training with reduced memory usage...")
            else:
                print(f"Error in training: {str(e)}")
                raise

# Create memory-optimized trainer
print("Creating optimized trainer for multi-GPU setup...")

# Custom callbacks for memory efficiency
callbacks = [
    early_stopping_callback,
    MemoryMonitorCallback(memory_threshold=0.9, check_interval=20)
]

# Create trainer with memory optimizations
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
    data_collator=data_collator,
    callbacks=callbacks,
    # Set maximum steps to train so we can abort if memory issues persist
    max_steps=max(500, len(tokenized_train) // (BATCH_SIZE * torch.cuda.device_count() * GRADIENT_ACCUMULATION_STEPS)),
)

print("Memory-optimized training setup complete")

# Final memory cleanup before training
cleanup_memory(aggressive=True)
print("Initial memory status before training:")
monitor_resources(show_memory_details=True)


# %%
# Function to monitor system resources during training
def monitor_resources():
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    mem = psutil.virtual_memory()
    cpu_percent = psutil.cpu_percent(interval=0.1)
    
    print(f"\nSystem Resources:")
    print(f"CPU Usage: {cpu_percent}%")
    print(f"Process Memory: {memory_info.rss / 1024 / 1024:.2f} MB")
    print(f"System Memory: {mem.percent}% used, {mem.available / 1024 / 1024:.2f} MB available\n")


# %%
# Run training with enhanced memory monitoring for multi-GPU setup
try:
    print("Starting training with memory-optimized configuration...")
    
    # Set critical PyTorch memory optimizations for multi-GPU training
    if torch.cuda.is_available():
        # Apply optimal settings for training
        if torch.cuda.device_count() > 1:
            print(f"Configuring PyTorch for {torch.cuda.device_count()} GPUs...")
            
            # Enable TF32 precision for faster training (on Ampere GPUs)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            # Use fraction of available memory (per GPU) - leave headroom
            gpu_memory_fraction = float(os.environ.get("CUDA_MEM_FRACTION", "0.85"))
            print(f"Setting GPU memory fraction to {gpu_memory_fraction}")
            for i in range(torch.cuda.device_count()):
                torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction, i)
                
            print("Multi-GPU optimizations applied")
        else:
            print("Single GPU detected. Optimizing for maximum utilization.")
            # For single GPU, we can use a slightly higher memory fraction
            torch.cuda.set_per_process_memory_fraction(0.9)
    
        # Apply additional memory optimizations
        if hasattr(torch.cuda, 'amp') and USE_FP16:
            print("Enabling AMP (Automatic Mixed Precision) for memory efficiency")
        
        # Reserve some memory by allocating and freeing to reduce fragmentation
        tmp_tensors = []
        try:
            for i in range(torch.cuda.device_count()):
                # Allocate and free a small tensor on each GPU to consolidate memory
                device = torch.device(f'cuda:{i}')
                tensor = torch.zeros((1024, 1024), device=device, dtype=torch.float16)
                tmp_tensors.append(tensor)
            # Now free them
            tmp_tensors = []
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"Warning: Memory pre-allocation failed: {e}")
            
    # Show memory status right before training
    print("\n===== PRE-TRAINING MEMORY STATUS =====")
    monitor_resources(show_memory_details=True)
    
    # Final aggressive memory cleanup before training
    print("\nPerforming final memory cleanup before training...")
    cleanup_memory(aggressive=True)
    
    # Start training
    start_time = time.time()
    
    # Use our safe training loop with OOM protection
    print("\n====== STARTING TRAINING WITH OOM PROTECTION ======")
    print(f"Batch size: {BATCH_SIZE} per GPU")
    print(f"Gradient accumulation steps: {GRADIENT_ACCUMULATION_STEPS}")
    print(f"Effective batch size: {BATCH_SIZE * (torch.cuda.device_count() if torch.cuda.is_available() else 1) * GRADIENT_ACCUMULATION_STEPS}")
    print(f"Training on {len(tokenized_train)} examples for {NUM_EPOCHS} epochs")
    
    # Run training with OOM protection
    train_result = safe_training_loop(trainer, max_retries=2)
    
    # Calculate and display training duration
    training_duration = time.time() - start_time
    hours, remainder = divmod(training_duration, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"\n====== TRAINING COMPLETED SUCCESSFULLY ======")
    print(f"Total training time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
    print(f"Training loss: {train_result.metrics['train_loss']:.4f}")
    
    # Show memory status after training
    print("\n===== POST-TRAINING MEMORY STATUS =====")
    monitor_resources(show_memory_details=True)
    
    # Clean up memory before saving
    cleanup_memory(aggressive=True)
    
    # Save the trained model
    print("\nSaving model...")
    save_path = "./phi3_swift_model"
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Model saved to {save_path} ({QUANT_BITS}-bit {quant_method} quantized)")
    
    # Save model configuration details
    with open("./phi3_swift_model/quantization_config.json", "w") as f:
        config_data = {
            "quantization_method": quant_method,
            "bits": QUANT_BITS,
            "lora_rank": LORA_R,
            "lora_alpha": LORA_ALPHA,
            "original_model": MODEL_NAME,
            "max_length": MAX_LENGTH
        }
        json.dump(config_data, f, indent=2)
    
    # Create appropriate loading instructions based on quantization method
    if USING_AQLM:
        loading_code = """```python
from aqlm import quantize
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("./phi3_swift_model")

# Load the base model first (to apply quantization)
base_model = AutoModelForCausalLM.from_pretrained("./phi3_swift_model")

# Apply AQLM quantization
model = quantize(base_model, bits=QUANT_BITS, lora_rank=LORA_R)
```"""
    else:
        loading_code = """```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("./phi3_swift_model")

# Configure 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

# Load the quantized model
model = AutoModelForCausalLM.from_pretrained(
    "./phi3_swift_model",
    quantization_config=bnb_config,
    device_map="auto"
)
```"""
        
    # Also save a README with information about the quantization
    with open("./phi3_swift_model/README.md", "w") as f:
        f.write(f"""# Phi-3-mini Quantized Model

This model is a {QUANT_BITS}-bit quantized version of `{MODEL_NAME}` trained for Swift programming.

## Quantization Details
- Method: {quant_method}
- Bits: {QUANT_BITS} 
- Training dataset: {DATASET_ID}
- Fine-tuning method: LoRA (Low-Rank Adaptation)
- LoRA rank: {LORA_R}
- LoRA alpha: {LORA_ALPHA}

## Usage

To load this model:

{loading_code}

This quantized model reduces memory usage significantly while maintaining most of the capabilities of the original model.
""")
    
    # Clean up memory
    cleanup_memory()
    
except RuntimeError as e:
    # Handle CUDA out of memory errors specifically
    if "CUDA out of memory" in str(e):
        print("\n\n==== FATAL ERROR: CUDA OUT OF MEMORY ====")
        print("The model is too large for your available GPU memory.")
        print("\nRecommended solutions:")
        print("1. Reduce MAX_LENGTH (currently", MAX_LENGTH, ")")
        print("2. Reduce BATCH_SIZE (currently", BATCH_SIZE, ")")
        print("3. Increase GRADIENT_ACCUMULATION_STEPS (currently", GRADIENT_ACCUMULATION_STEPS, ")")
        print("4. Use more GPUs if available")
        print(f"5. Try a different device_map strategy (currently '{DEVICE_MAP_STRATEGY}')")
        print("\nDetailed error:")
        print(str(e))
    else:
        # Handle other runtime errors
        print(f"\n\n==== RUNTIME ERROR DURING TRAINING ====")
        print(str(e))
        traceback.print_exc()
        
    # Monitor resources after error
    print("\n===== MEMORY STATUS AFTER ERROR =====")
    monitor_resources(show_memory_details=True)

except Exception as e:
    # Handle all other exceptions
    print(f"\n\n==== UNEXPECTED ERROR DURING TRAINING ====")
    print(str(e))
    traceback.print_exc()
    
    # Monitor resources after error
    print("\n===== MEMORY STATUS AFTER ERROR =====")
    monitor_resources(show_memory_details=True)

finally:
    # Always perform cleanup, even if there was an error
    print("\nPerforming final cleanup...")
    cleanup_memory(aggressive=True)
    
    # Report total execution time
    total_duration = time.time() - start_time
    hours, remainder = divmod(total_duration, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Total execution time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
    
    # Helpful message about retrying with different settings if needed
    if 'train_result' not in locals():
        print("\n⚠️ Training was not successful.")
        print("If you continue to experience memory issues, try editing the script with these changes:")
        print("- Set MAX_LENGTH to 1024 (currently", MAX_LENGTH, ")")
        print("- Set BATCH_SIZE to 1 (currently", BATCH_SIZE, ")")
        print("- Set GRADIENT_ACCUMULATION_STEPS to 16 (currently", GRADIENT_ACCUMULATION_STEPS, ")")
        print("- Set LORA_R to 4 (currently", LORA_R, ")")
    else:
        print("\nIf you want to use the model for inference, load it with:")
        print(f"model = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained('{MODEL_NAME}'), '{save_path}')")
        print(f"tokenizer = AutoTokenizer.from_pretrained('{save_path}')")
        print("\nFor multi-GPU inference, use device_map='auto'")

# %%
# Test the model with Swift code examples
try:
    print(f"Testing the {QUANT_BITS}-bit {quant_method} quantized model with Swift code examples...")
    
    # For testing, we use the model we already have loaded
    test_model = model
    
    # Function to generate responses for test examples
    def generate_response(prompt):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            # Generate with the quantized model
            outputs = test_model.generate(
                inputs.input_ids,
                max_new_tokens=200,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract just the assistant's response
        if "<|assistant|>" in response:
            response = response.split("<|assistant|>")[-1].strip()
        return response
    
    # Test prompts for different Swift language tasks
    test_prompts = [
        # Explain Swift syntax
        "<|user|>\nExplain the key features of Swift's optional unwrapping syntax:\n\n```swift\nfunc processName(_ name: String?) {\n    guard let unwrappedName = name else {\n        print(\"No name provided\")\n        return\n    }\n    print(\"Hello, \\(unwrappedName)!\")\n}\n```\n<|assistant|>",
        
        # Code completion
        "<|user|>\nComplete this Swift function that calculates the factorial of a number:\n\n```swift\nfunc factorial(_ n: Int) -> Int {\n    // Add implementation here\n}\n```\n<|assistant|>",
        
        # Debugging help
        "<|user|>\nWhat's wrong with this Swift code and how can I fix it?\n\n```swift\nclass Person {\n    var name: String\n    var age: Int\n    \n    func greet() {\n        print(\"Hello, my name is \\(name) and I am \\(age) years old.\")\n    }\n}\n\nlet person = Person()\nperson.greet()\n```\n<|assistant|>",
        
        # Swift best practices
        "<|user|>\nExplain Swift best practices for error handling:\n<|assistant|>"
    ]
    
    # Generate and print responses
    for i, prompt in enumerate(test_prompts):
        print(f"\nTest {i+1}:\n{'-'*40}")
        print(f"Prompt: {prompt.split('<|assistant|>')[0].replace('<|user|>', '')}")
        response = generate_response(prompt)
        print(f"\nResponse:\n{response}\n")
    
    print("\nTesting complete")
except Exception as e:
    print(f"Error during testing: {e}")
    import traceback
    traceback.print_exc()
