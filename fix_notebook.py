import json
import re

# Load the notebook
with open('phi-train.ipynb', 'r') as f:
    notebook_content = f.read()

# The notebook has a nested structure - it's a notebook containing a single cell with a JSON string
# that represents another notebook. We need to modify the JSON string directly.

# 1. First, let's add padding=True and truncation=True to the data_collator
# We'll use regex to find and modify the data_collator definition
data_collator_pattern = r'(data_collator\s*=\s*[^=\n]*\()'
data_collator_replacement = r'\1padding=True, truncation=True, '
modified_content = re.sub(data_collator_pattern, data_collator_replacement, notebook_content)

# 2. Add label_names=[] to the Trainer
trainer_pattern = r'(Trainer\s*\()([^\)]*)'
# Check if label_names is already in the trainer args
if 'label_names' not in notebook_content:
    # Find all Trainer instances
    trainer_matches = re.finditer(trainer_pattern, modified_content)
    for match in trainer_matches:
        # Get the full match and the arguments
        full_match = match.group(0)
        args = match.group(2)
        
        # Add label_names=[] to the arguments
        if args.strip():  # If there are already arguments
            # Add label_names after the first argument
            new_args = args.rstrip() + ',\n    label_names=[]'
        else:  # If there are no arguments yet
            new_args = 'label_names=[]'
        
        # Replace the original Trainer call with the modified one
        modified_content = modified_content.replace(full_match, f'Trainer({new_args}')

# Save the modified notebook
with open('phi-train-fixed.ipynb', 'w') as f:
    f.write(modified_content)

print("Notebook fixed and saved to phi-train-fixed.ipynb")