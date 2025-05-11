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
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # Explicitly set to use 2 GPUs

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
# Set up training arguments with optimized settings for multi-GPU training
try:
    # Create output directory if it doesn't exist
    os.makedirs("./phi3_swift_model", exist_ok=True)
    
    # Configure training arguments with distributed training settings
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
        save_steps=500,
        save_total_limit=2,
        eval_strategy="steps",
        eval_steps=500,
        load_best_model_at_end=True,
        fp16=True,  # Use mixed precision training
        gradient_checkpointing=True,  # Enable gradient checkpointing to save memory
        # Distributed training parameters
        local_rank=int(os.environ.get("LOCAL_RANK", -1)),  # For distributed training
        ddp_find_unused_parameters=False,  # Optimize DDP
        dataloader_num_workers=4,  # Parallelize data loading
        report_to="none",  # Disable reporting to avoid extra overhead
    )
    
    print(f"Training arguments configured for {'multi-GPU' if torch.cuda.device_count() > 1 else 'single-GPU'} training")
    print(f"Using gradient checkpointing: {training_args.gradient_checkpointing}")
    print(f"Using mixed precision: {training_args.fp16}")
    print(f"Local rank: {training_args.local_rank}")
    
except Exception as e:
    print(f"Error setting up training arguments: {e}")
    raise

# %%
# Define early stopping callback
early_stopping_callback = EarlyStoppingCallback(
    early_stopping_patience=3,
    early_stopping_threshold=0.01
)


# %%
# FIXED: Create a custom data collator that properly handles the data
class CustomDataCollatorForLanguageModeling(DataCollatorForLanguageModeling):
    def __call__(self, features):
        # Ensure all features have the same keys
        if not all(k in features[0] for k in ["input_ids", "attention_mask", "labels"]):
            raise ValueError("Some features are missing required keys")
        
        # Create a batch with proper padding
        batch = {
            "input_ids": torch.stack([f["input_ids"] for f in features]),
            "attention_mask": torch.stack([f["attention_mask"] for f in features]),
            "labels": torch.stack([f["labels"] for f in features])
        }
        
        return batch

# Create data collator for language modeling
data_collator = CustomDataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # We're doing causal language modeling, not masked language modeling
)

# %%
# Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
    data_collator=data_collator,
    callbacks=[early_stopping_callback]
)

print("Training setup complete")


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
    print("Starting training...")
    
    # Monitor resources before training
    print("Resources before training:")
    monitor_resources()
    
    # Additional memory cleanup before training
    cleanup_memory()
    
    # Set PyTorch to optimize for multi-GPU training
    if torch.cuda.device_count() > 1:
        print("Configuring PyTorch for multi-GPU training...")
        # Enable TF32 precision for faster training (on Ampere GPUs)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Set memory allocation strategy
        torch.cuda.set_per_process_memory_fraction(0.95)  # Reserve some memory for system
    
    # Start training with a timeout
    start_time = time.time()
    
    # Run training
    train_result = trainer.train()
    
    # Monitor resources after training
    print("Resources after training:")
    monitor_resources()
    
    # Print training results
    print(f"Training completed in {train_result.metrics['train_runtime']:.2f} seconds")
    print(f"Training loss: {train_result.metrics['train_loss']:.4f}")
    
    # Save the model
    trainer.save_model("./phi3_swift_model")
    print("Model saved to ./phi3_swift_model")
    
    # Clean up memory
    cleanup_memory()
    
except Exception as e:
    print(f"Error during training: {e}")
    
    # Print stack trace for debugging
    import traceback
    traceback.print_exc()
    
    # Monitor resources after error
    print("Resources after error:")
    monitor_resources()
    
    raise

# %%
# Comprehensive evaluation to measure how much the model has learned
print("\n" + "="*80)
print("QUANTITATIVE EVALUATION OF MODEL LEARNING")
print("="*80)

try:
    # 1. Load original pre-trained model for comparison
    print("\n[1] Loading original pre-trained model for comparison...")
    
    # Use the same quantization config for fair comparison
    original_bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )
    
    try:
        # Load the original model with the same quantization as our fine-tuned model
        original_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,  # Use the original model name
            quantization_config=original_bnb_config,
            device_map="auto" if torch.cuda.is_available() else None,
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        print(f"Successfully loaded original model: {MODEL_NAME}")
        has_original_model = True
    except Exception as e:
        print(f"Warning: Could not load original model for comparison: {e}")
        print("Will proceed with evaluations that don't require the original model.")
        has_original_model = False
    
    # 2. Calculate perplexity on test set
    print("\n[2] Calculating perplexity on test set...")
    
    def calculate_perplexity(eval_model, eval_dataset, batch_size=4):
        """Calculate perplexity on evaluation dataset."""
        eval_dataloader = torch.utils.data.DataLoader(
            eval_dataset, batch_size=batch_size, shuffle=False
        )
        
        eval_model.eval()
        total_loss = 0
        total_tokens = 0
        
        with torch.no_grad():
            for batch in eval_dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = eval_model(**batch)
                loss = outputs.loss
                
                # Get the number of tokens in the batch
                num_tokens = batch["attention_mask"].sum().item()
                
                total_loss += loss.item() * num_tokens
                total_tokens += num_tokens
        
        # Calculate perplexity
        avg_loss = total_loss / total_tokens
        perplexity = np.exp(avg_loss)
        
        return perplexity
    
    # Calculate perplexity for fine-tuned model
    fine_tuned_perplexity = calculate_perplexity(model, tokenized_val)
    print(f"Fine-tuned model perplexity: {fine_tuned_perplexity:.4f}")
    
    if has_original_model:
        # Calculate perplexity for original model if available
        original_perplexity = calculate_perplexity(original_model, tokenized_val)
        print(f"Original model perplexity: {original_perplexity:.4f}")
        
        # Calculate improvement
        if original_perplexity > fine_tuned_perplexity:
            improvement = ((original_perplexity - fine_tuned_perplexity) / original_perplexity) * 100
            print(f"Perplexity improvement: {improvement:.2f}% better than original model")
        else:
            decline = ((fine_tuned_perplexity - original_perplexity) / original_perplexity) * 100
            print(f"Perplexity decline: {decline:.2f}% worse than original model")
    
    # 3. Function to generate responses for evaluation
    def generate_response(eval_model, prompt, max_tokens=200):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = eval_model.generate(
                inputs.input_ids,
                max_new_tokens=max_tokens,
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
    
    # 4. Define a structured benchmark for Swift programming tasks
    print("\n[3] Executing Swift programming benchmark...")
    
    swift_benchmark = [
        {
            "name": "Optional Unwrapping",
            "category": "Language Fundamentals",
            "prompt": "<|user|>\nExplain the key features of Swift's optional unwrapping syntax:\n\n```swift\nfunc processName(_ name: String?) {\n    guard let unwrappedName = name else {\n        print(\"No name provided\")\n        return\n    }\n    print(\"Hello, \\(unwrappedName)!\")\n}\n```\n<|assistant|>",
            "keywords": ["guard", "optional", "unwrap", "if let", "nil", "safety"]
        },
        {
            "name": "Factorial Implementation",
            "category": "Algorithm Implementation",
            "prompt": "<|user|>\nComplete this Swift function that calculates the factorial of a number:\n\n```swift\nfunc factorial(_ n: Int) -> Int {\n    // Add implementation here\n}\n```\n<|assistant|>",
            "keywords": ["recursion", "base case", "return 1", "multiply", "n * factorial"]
        },
        {
            "name": "Class Initialization",
            "category": "Object-Oriented Programming",
            "prompt": "<|user|>\nWhat's wrong with this Swift code and how can I fix it?\n\n```swift\nclass Person {\n    var name: String\n    var age: Int\n    \n    func greet() {\n        print(\"Hello, my name is \\(name) and I am \\(age) years old.\")\n    }\n}\n\nlet person = Person()\nperson.greet()\n```\n<|assistant|>",
            "keywords": ["initializer", "init", "required", "properties", "constructor"]
        },
        {
            "name": "Error Handling Patterns",
            "category": "Best Practices",
            "prompt": "<|user|>\nExplain Swift best practices for error handling:\n<|assistant|>",
            "keywords": ["throws", "do-catch", "try", "Error protocol", "Result type"]
        },
        {
            "name": "Protocol Implementation",
            "category": "Swift Protocols",
            "prompt": "<|user|>\nExplain how to use protocols in Swift and provide an example of protocol conformance:\n<|assistant|>",
            "keywords": ["protocol", "conform", "implement", "requirement", "extension"]
        }
    ]
    
    # 5. Evaluate and score responses based on keyword presence
    def score_response(response, keywords):
        """Score a response based on the presence of expected keywords."""
        score = 0
        matched_keywords = []
        
        for keyword in keywords:
            if keyword.lower() in response.lower():
                score += 1
                matched_keywords.append(keyword)
        
        percentage = (score / len(keywords)) * 100 if keywords else 0
        return {
            "score": score,
            "total": len(keywords),
            "percentage": percentage,
            "matched_keywords": matched_keywords
        }
    
    # 6. Run the benchmark and collect results
    benchmark_results = []
    
    for test in swift_benchmark:
        print(f"\nEvaluating: {test['name']} ({test['category']})")
        
        # Generate response with fine-tuned model
        fine_tuned_response = generate_response(model, test['prompt'])
        fine_tuned_score = score_response(fine_tuned_response, test['keywords'])
        
        result = {
            "name": test['name'],
            "category": test['category'],
            "fine_tuned_score": fine_tuned_score
        }
        
        if has_original_model:
            # Generate response with original model if available
            original_response = generate_response(original_model, test['prompt'])
            original_score = score_response(original_response, test['keywords'])
            result["original_score"] = original_score
            
            # Calculate improvement
            score_diff = fine_tuned_score["percentage"] - original_score["percentage"]
            result["improvement"] = score_diff
            
            print(f"  Original model score: {original_score['percentage']:.1f}%")
            print(f"  Fine-tuned model score: {fine_tuned_score['percentage']:.1f}%")
            print(f"  Improvement: {score_diff:+.1f}%")
        else:
            print(f"  Fine-tuned model score: {fine_tuned_score['percentage']:.1f}%")
        
        benchmark_results.append(result)
    
    # 7. Calculate overall learning metrics
    print("\n[4] Calculating overall learning metrics...")
    
    # Average score across all benchmark tests
    avg_fine_tuned_score = sum(r["fine_tuned_score"]["percentage"] for r in benchmark_results) / len(benchmark_results)
    print(f"Average fine-tuned model score: {avg_fine_tuned_score:.2f}%")
    
    if has_original_model:
        avg_original_score = sum(r["original_score"]["percentage"] for r in benchmark_results) / len(benchmark_results)
        avg_improvement = sum(r["improvement"] for r in benchmark_results) / len(benchmark_results)
        print(f"Average original model score: {avg_original_score:.2f}%")
        print(f"Average improvement: {avg_improvement:+.2f}%")
    
    # 8. Generate summary report
    print("\n" + "="*80)
    print("LEARNING ASSESSMENT SUMMARY")
    print("="*80)
    
    print("\nPerformance by category:")
    categories = set(test["category"] for test in swift_benchmark)
    for category in categories:
        category_tests = [r for r in benchmark_results if r["category"] == category]
        avg_category_score = sum(r["fine_tuned_score"]["percentage"] for r in category_tests) / len(category_tests)
        print(f"  - {category}: {avg_category_score:.2f}%")
    
    # Learning assessment based on scores
    if avg_fine_tuned_score > 80:
        learning_assessment = "Excellent"
    elif avg_fine_tuned_score > 60:
        learning_assessment = "Good"
    elif avg_fine_tuned_score > 40:
        learning_assessment = "Moderate"
    else:
        learning_assessment = "Limited"
    
    print(f"\nOverall learning assessment: {learning_assessment}")
    
    if has_original_model:
        improvement_assessment = ""
        if avg_improvement > 20:
            improvement_assessment = "Substantial improvement over the original model"
        elif avg_improvement > 10:
            improvement_assessment = "Significant improvement over the original model"
        elif avg_improvement > 0:
            improvement_assessment = "Modest improvement over the original model"
        else:
            improvement_assessment = "No improvement over the original model"
        
        print(f"Comparative assessment: {improvement_assessment}")
    
    # 9. Save evaluation results
    evaluation_results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_name": MODEL_NAME,
        "fine_tuned_perplexity": float(fine_tuned_perplexity),
        "benchmark_results": [
            {
                "name": r["name"],
                "category": r["category"],
                "fine_tuned_score": {
                    "percentage": float(r["fine_tuned_score"]["percentage"]),
                    "score": r["fine_tuned_score"]["score"],
                    "total": r["fine_tuned_score"]["total"]
                }
            } for r in benchmark_results
        ],
        "average_score": float(avg_fine_tuned_score),
        "learning_assessment": learning_assessment
    }
    
    if has_original_model:
        evaluation_results["original_perplexity"] = float(original_perplexity)
        evaluation_results["perplexity_improvement"] = float(improvement) if "improvement" in locals() else float(-decline)
        for i, r in enumerate(evaluation_results["benchmark_results"]):
            r["original_score"] = {
                "percentage": float(benchmark_results[i]["original_score"]["percentage"]),
                "score": benchmark_results[i]["original_score"]["score"],
                "total": benchmark_results[i]["original_score"]["total"]
            }
            r["improvement"] = float(benchmark_results[i]["improvement"])
        
        evaluation_results["average_original_score"] = float(avg_original_score)
        evaluation_results["average_improvement"] = float(avg_improvement)
        evaluation_results["improvement_assessment"] = improvement_assessment
    
    # Save to file
    evaluation_file = "./phi3_swift_model/evaluation_results.json"
    with open(evaluation_file, "w") as f:
        json.dump(evaluation_results, f, indent=2)
    
    print(f"\nEvaluation results saved to {evaluation_file}")
    
    # 10. Visual qualitative comparison (display a few examples)
    print("\n" + "="*80)
    print("QUALITATIVE COMPARISON EXAMPLES")
    print("="*80)
    
    # Choose a couple of tests for detailed comparison
    detailed_examples = swift_benchmark[:2]  # Just use the first two tests for brevity
    
    for i, test in enumerate(detailed_examples):
        print(f"\nExample {i+1}: {test['name']}")
        print("-" * 40)
        
        print(f"Prompt: {test['prompt'].split('<|assistant|>')[0].replace('<|user|>', '')}")
        
        fine_tuned_response = generate_response(model, test['prompt'])
        print(f"\nFine-tuned model response:")
        print(fine_tuned_response[:500] + ("..." if len(fine_tuned_response) > 500 else ""))
        
        if has_original_model:
            original_response = generate_response(original_model, test['prompt'])
            print(f"\nOriginal model response:")
            print(original_response[:500] + ("..." if len(original_response) > 500 else ""))
    
    print("\nEvaluation complete!\n")

except Exception as e:
    print(f"Error during evaluation: {e}")
    import traceback
    traceback.print_exc()

# Original simple test examples remain but are now supplementary
print("\n" + "="*80)
print("SUPPLEMENTARY TEST EXAMPLES")
print("="*80)

try:
    print("Testing the model with Swift code examples...")
    
    # Function to generate responses for test examples
    def generate_response(prompt):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
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
