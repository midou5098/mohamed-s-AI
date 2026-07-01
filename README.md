# Mohamed's AI

Fine-tuning artifacts for a personal AI assistant trained to mimic Mohamed's writing style and personality.

This repository is based on a LoRA fine-tune of [`unsloth/qwen2.5-3b-bnb-4bit`](https://huggingface.co/unsloth/qwen2.5-3b-bnb-4bit), a 4-bit Qwen2.5 3B causal language model. The adapter configuration targets the main attention and MLP projection layers, making it lightweight compared with a full model checkpoint.

## What is included

| File | Purpose |
| --- | --- |
| `adapter_config.json` | PEFT/LoRA adapter configuration for the fine-tuned model. |
| `tokenizer.json` | Qwen2 tokenizer data. |
| `tokenizer_config.json` | Tokenizer settings, including max context length and padding token. |
| `trainer_state.json` | Training progress, logging history, and final training metadata. |
| `training_args.bin` | Serialized Hugging Face training arguments. |
| `optimizer.pt` | Optimizer checkpoint from training. |
| `scheduler.pt` | Learning-rate scheduler checkpoint. |
| `scaler.pt` | Mixed-precision scaler state. |
| `rng_state.pth` | Random number generator state for reproducibility/resuming. |

## Current status

The training run completed:

- Base model: `unsloth/qwen2.5-3b-bnb-4bit`
- Model class: `Qwen2ForCausalLM`
- Fine-tuning method: LoRA
- Task type: causal language modeling
- Epochs: `3`
- Max steps: `1500`
- Train batch size: `2`
- Save interval: every `500` steps
- Logging interval: every `10` steps
- LoRA rank: `16`
- LoRA alpha: `16`
- LoRA dropout: `0`
- Tokenizer max length: `32768`

Important: this repository currently contains `adapter_config.json`, but it does **not** contain adapter weights such as `adapter_model.safetensors` or `adapter_model.bin`. Those weights are required to load the fine-tuned personality adapter for inference. Without them, the repo is useful as training metadata/checkpoint context, but not as a complete runnable model.

## Expected complete model layout

A complete PEFT adapter repository normally includes at least:

```text
adapter_config.json
adapter_model.safetensors
tokenizer.json
tokenizer_config.json
README.md
```

If this model is pushed to GitHub, large binary files should usually be tracked with Git LFS.

## Installation

Create a Python environment and install the common inference dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch transformers peft accelerate bitsandbytes
```

Depending on your GPU and CUDA setup, you may need to install the PyTorch build that matches your system.

## Inference example

After adding the adapter weights, the model can be loaded with Transformers and PEFT:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

base_model = "unsloth/qwen2.5-3b-bnb-4bit"
adapter_path = "."

tokenizer = AutoTokenizer.from_pretrained(adapter_path)
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    device_map="auto",
    torch_dtype=torch.float16,
)
model = PeftModel.from_pretrained(model, adapter_path)

prompt = "Write a short message in Mohamed's style:"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=150,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
    )

print(tokenizer.decode(output[0], skip_special_tokens=True))
```

## Training summary

The training logs in `trainer_state.json` show loss decreasing over the run, from about `3.11` near step `10` to about `1.44` at step `1500`. The learning rate warmed up near the start and then decayed toward zero by the final step.

No evaluation metric or best checkpoint is recorded in the trainer state, so the model should be manually tested before release.

## Recommended next steps

1. Add the missing adapter weights file, preferably `adapter_model.safetensors`.
2. Add a short example prompt and response once the model output is verified.
3. Add license information for both this adapter and the base model dependency.
4. Consider publishing the adapter to Hugging Face if the goal is easy reuse with `from_pretrained`.

## why i built this 
somewhy felt i wanted to test how it felt to talk to myslf , so this repo exists merly bcz of human connection
