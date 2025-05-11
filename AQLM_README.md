# AQLM 2-bit Quantization for Phi-3 Model

This repository implements AQLM (Activation-aware Quantized Language Model) 2-bit quantization for the Phi-3-mini-128k-instruct model.

## Benefits

- **Memory Reduction**: 50% memory savings compared to 4-bit quantization
- **Performance Preservation**: Better accuracy at ultra-low bit depths
- **Multi-GPU Support**: Automatic distribution across available GPUs
- **Offloading**: Support for offloading to CPU/disk for larger models

## Implementation

The implementation uses the AQLM library to configure 2-bit quantization:

```python
aqlm_config = AqlmConfig(
    bits=2,                        # Use 2-bit quantization
    device_map="auto",             # Automatically distribute model across available GPUs
    max_memory=None,               # Use maximum available memory
    offload_folder="aqlm_offload", # Folder for offloading to disk if needed
    trust_remote_code=True,        # Trust remote code for model loading
    dtype="float16"                # Use float16 for remaining parameters
)
```

This enables more efficient training of the Phi-3 model on Swift code.
