"""
Script: scripts/generate_colab_notebooks.py
Generates train_sft.ipynb and train_dpo.ipynb with full Colab metadata,
markdown documentation, and executable code cells.
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def make_notebook(cells):
    return {
        "nbformat": 4,
        "nbformat_minor": 0,
        "metadata": {
            "colab": {
                "name": "Colab Notebook",
                "provenance": [],
                "gpuType": "T4",
            },
            "kernelspec": {
                "display_name": "Python 3",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            },
            "accelerator": "GPU"
        },
        "cells": cells
    }


def markdown_cell(source):
    lines = [line + "\n" for line in source.strip().split("\n")]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": lines
    }


def code_cell(source):
    lines = [line + "\n" for line in source.strip().split("\n")]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines
    }


def build_sft_notebook():
    cells = [
        markdown_cell("""# 🎓 Supervised Fine-Tuning (SFT) with QLoRA for Educational RAG
### Model: `meta-llama/Meta-Llama-3-8B-Instruct`
### Frameworks: `transformers`, `peft`, `trl`, `bitsandbytes`, `accelerate`

---
### 📌 Overview
This notebook implements **4-bit Parameter-Efficient Fine-Tuning (QLoRA)** using Hugging Face's `trl.SFTTrainer`.
- **Target**: Fine-tunes `Meta-Llama-3-8B-Instruct` on educational Q&A data.
- **Hardware Requirement**: Runs on a **free Google Colab T4 GPU (15 GB VRAM)** or higher (A100, V100, L4).
- **Features**:
  - 4-bit NormalFloat (NF4) quantization with double quantization.
  - LoRA adapters on all attention and MLP projection layers (`q, k, v, o, gate, up, down`).
  - Llama-3 conversational prompt formatting (`apply_chat_template`).
  - Exports the trained LoRA adapter ready for Ollama or GGUF conversion."""),

        code_cell("""# Step 1: Install Required Libraries in Colab
!pip install -q --upgrade pip
!pip install -q transformers>=4.40.0 trl>=0.8.6 peft>=0.10.0 bitsandbytes>=0.43.0 accelerate>=0.29.0 datasets>=2.18.0"""),

        code_cell("""# Step 2: Verify GPU Acceleration
import torch
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")
    print(f"GPU VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
else:
    raise RuntimeError("Please enable GPU acceleration in Colab: Runtime -> Change runtime type -> T4 GPU")"""),

        code_cell("""# Step 3: Hugging Face Authentication
# Meta-Llama-3-8B-Instruct is a gated model. You must accept the license on Hugging Face
# and log in with your read-access Hugging Face token.
from huggingface_hub import login
from google.colab import userdata

# Option A: Colab Secrets (recommended)
try:
    hf_token = userdata.get("HF_TOKEN")
    login(token=hf_token)
    print("Logged in to Hugging Face via Colab Secret 'HF_TOKEN'!")
except Exception:
    # Option B: Interactive Login
    print("Please enter your Hugging Face Token (with read permission):")
    login()"""),

        code_cell("""# Step 4: Create Sample Educational SFT Dataset (JSONL)
import json

sft_data = [
    {
        "messages": [
            {
                "role": "system",
                "content": "You are an expert educational AI tutor. Answer questions factually and strictly based on the provided context with citations."
            },
            {
                "role": "user",
                "content": "Context Chunks:\\n[1] Isaac Newton published the Philosophiæ Naturalis Principia Mathematica in 1687, formulating the laws of motion and universal gravitation.\\n\\nQuestion: When and in what book did Newton introduce his laws of motion?"
            },
            {
                "role": "assistant",
                "content": "Sir Isaac Newton introduced his laws of motion in his foundational work, *Philosophiæ Naturalis Principia Mathematica*, published in 1687 [1]."
            }
        ]
    },
    {
        "messages": [
            {
                "role": "system",
                "content": "You are an expert educational AI tutor. Answer questions factually and strictly based on the provided context with citations."
            },
            {
                "role": "user",
                "content": "Context Chunks:\\n[1] Newton's First Law (Law of Inertia) states that an object remains at rest or in uniform motion unless acted upon by a net external force.\\n[2] Newton's Second Law defines force as F = ma.\\n\\nQuestion: What does Newton's First Law state?"
            },
            {
                "role": "assistant",
                "content": "Newton's First Law, also known as the Law of Inertia, states that an object will continue in its state of rest or uniform motion in a straight line unless it is compelled to change that state by an external net force [1]."
            }
        ]
    },
    {
        "messages": [
            {
                "role": "system",
                "content": "You are an expert educational AI tutor. Answer questions factually and strictly based on the provided context with citations."
            },
            {
                "role": "user",
                "content": "Context Chunks:\\n[1] Newton's Third Law states that whenever one body exerts a force on a second body, the second body exerts an equal and opposite force on the first body.\\n\\nQuestion: Explain action-reaction forces in Newton's Third Law."
            },
            {
                "role": "assistant",
                "content": "According to Newton's Third Law, forces always occur in matched pairs: whenever an object exerts a force (action) on another, the second object simultaneously exerts an equal magnitude force in the opposite direction (reaction) on the first [1]."
            }
        ]
    },
    {
        "messages": [
            {
                "role": "system",
                "content": "You are an expert educational AI tutor. Answer questions factually and strictly based on the provided context with citations."
            },
            {
                "role": "user",
                "content": "Context Chunks:\\n[1] Photosynthesis is the process by which green plants transform light energy into chemical energy stored in glucose.\\n\\nQuestion: Who won the 1982 football World Cup?"
            },
            {
                "role": "assistant",
                "content": "I am sorry, but the provided context does not contain sufficient educational information to answer this question."
            }
        ]
    }
]

# Write to sft_dataset.jsonl
with open("sft_dataset.jsonl", "w", encoding="utf-8") as f:
    for item in sft_data:
        f.write(json.dumps(item) + "\\n")

print(f"Created sft_dataset.jsonl with {len(sft_data)} educational training examples.")"""),

        code_cell("""# Step 5: Configure 4-bit QLoRA Quantization
import torch
from transformers import BitsAndBytesConfig

compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",           # 4-bit NormalFloat for optimal weight distribution
    bnb_4bit_compute_dtype=compute_dtype, # bfloat16 or float16 for fast matrix ops
    bnb_4bit_use_double_quant=True,       # Nested quantization saving ~0.4 bits/param
)
print("BitsAndBytes 4-bit config successfully initialized.")"""),

        code_cell("""# Step 6: Load Tokenizer and Base Model
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import prepare_model_for_kbit_training

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

print(f"Loading Tokenizer for {model_id}...")
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

print(f"Loading 4-bit Quantized Base Model {model_id}...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=compute_dtype,
    trust_remote_code=True,
)

# Prepare model for LoRA training (freezes base model, enables gradient checkpointing)
model = prepare_model_for_kbit_training(model)
model.config.use_cache = False  # Disabled during training for gradient checkpointing
print("Base model loaded into VRAM successfully!")"""),

        code_cell("""# Step 7: Configure LoRA Adapter (PEFT)
from peft import LoraConfig, get_peft_model

peft_config = LoraConfig(
    r=16,                              # LoRA rank dimension
    lora_alpha=32,                     # Scaling factor (typically 2 * r)
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",   # Attention projections
        "gate_proj", "up_proj", "down_proj"       # MLP feed-forward projections
    ],
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()"""),

        code_cell("""# Step 8: Load and Format the Dataset with Llama-3 Chat Template
from datasets import load_dataset

dataset = load_dataset("json", data_files="sft_dataset.jsonl", split="train")

def format_conversations(batch):
    formatted_texts = []
    for messages in batch["messages"]:
        # Applies official <|begin_of_text|><|start_header_id|>... Llama 3 prompt structure
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        formatted_texts.append(text)
    return {"text": formatted_texts}

dataset = dataset.map(format_conversations, batched=True)
print("Formatted sample text preview:\\n")
print(dataset[0]["text"][:400] + "...\\n")"""),

        code_cell("""# Step 9: Configure and Launch SFTTrainer
from trl import SFTTrainer, SFTConfig

training_args = SFTConfig(
    output_dir="./sft_output",
    dataset_text_field="text",
    max_seq_length=512,
    per_device_train_batch_size=1,       # 1 or 2 fits easily on 15GB T4 GPU
    gradient_accumulation_steps=4,       # Effective batch size = 4
    optim="paged_adamw_8bit",            # Memory-efficient 8-bit optimizer
    learning_rate=2e-4,                  # Standard QLoRA learning rate
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    logging_steps=1,
    num_train_epochs=3,                  # Adjust based on real dataset size
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    save_strategy="epoch",
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    tokenizer=tokenizer,
    args=training_args,
)

print("Starting Supervised Fine-Tuning...")
trainer.train()
print("SFT Training Completed Successfully!")"""),

        code_cell("""# Step 10: Save Fine-Tuned LoRA Adapter
adapter_save_dir = "./educational_sft_adapter"

trainer.model.save_pretrained(adapter_save_dir)
tokenizer.save_pretrained(adapter_save_dir)

print(f"LoRA Adapter saved to '{adapter_save_dir}'!")
!ls -lh {adapter_save_dir}"""),

        code_cell("""# Step 11: Inference Test on Fine-Tuned Model
from transformers import pipeline

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=150,
    temperature=0.2,
    do_sample=True,
    top_p=0.9,
)

test_messages = [
    {
        "role": "system",
        "content": "You are an expert educational AI tutor. Answer questions factually and strictly based on the provided context with citations."
    },
    {
        "role": "user",
        "content": "Context Chunks:\\n[1] Isaac Newton published the Principia in 1687.\\n\\nQuestion: When was Newton's Principia published?"
    }
]

prompt = tokenizer.apply_chat_template(test_messages, tokenize=False, add_generation_prompt=True)
outputs = pipe(prompt)
generated_text = outputs[0]["generated_text"]
print("Model Response:\\n")
print(generated_text[len(prompt):])""")
    ]

    return make_notebook(cells)


def build_dpo_notebook():
    cells = [
        markdown_cell("""# ⚖️ Direct Preference Optimization (DPO) for Educational RAG
### Aligning `Meta-Llama-3-8B-Instruct` for Factual Grounding & Hallucination Refusal
### Frameworks: `trl.DPOTrainer`, `peft`, `transformers`, `bitsandbytes`

---
### 📌 Overview
This notebook trains an educational LLM using **Direct Preference Optimization (DPO)** via TRL's `DPOTrainer`.
- **Why DPO?**: While SFT teaches the model *how to speak*, DPO teaches the model *what to prefer*:
  - **Chosen Response**: Strict factual adherence, accurate citations, and proper refusal when context is missing.
  - **Rejected Response**: Hallucinations, external knowledge injection, or missing citations.
- **Hardware Requirement**: Runs on a **Google Colab T4 GPU (15 GB VRAM)** with 4-bit QLoRA.
- **Reference Model**: When using PEFT/LoRA, `DPOTrainer` uses the base model weights with LoRA disabled as the implicit frozen reference policy!"""),

        code_cell("""# Step 1: Install Required Libraries in Colab
!pip install -q --upgrade pip
!pip install -q transformers>=4.40.0 trl>=0.8.6 peft>=0.10.0 bitsandbytes>=0.43.0 accelerate>=0.29.0 datasets>=2.18.0"""),

        code_cell("""# Step 2: Verify GPU Acceleration
import torch
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB VRAM)")
else:
    raise RuntimeError("Please enable GPU acceleration in Colab: Runtime -> Change runtime type -> T4 GPU")"""),

        code_cell("""# Step 3: Hugging Face Authentication
from huggingface_hub import login
from google.colab import userdata

try:
    hf_token = userdata.get("HF_TOKEN")
    login(token=hf_token)
    print("Logged in to Hugging Face via Colab Secret!")
except Exception:
    login()"""),

        code_cell("""# Step 4: Create Educational Preference Dataset (Chosen vs. Rejected)
import json

dpo_data = [
    {
        # Example 1: Grounded answer vs. Hallucinated answer
        "prompt": "<|start_header_id|>system<|end_header_id|>\\nYou are an expert educational AI tutor. Answer questions factually and strictly based on the provided context with citations.<|eot_id|><|start_header_id|>user<|end_header_id|>\\nContext Chunks:\\n[1] Isaac Newton published the Principia in 1687, presenting the three laws of motion.\\n\\nQuestion: When was the Principia published?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\\n",
        "chosen": "According to the provided context, the Principia was published in 1687 by Isaac Newton [1].",
        "rejected": "The Principia was written by Albert Einstein in 1905, establishing modern general relativity and quantum mechanics."
    },
    {
        # Example 2: Out-of-Domain Query -> Proper Refusal vs. Speculative Hallucination
        "prompt": "<|start_header_id|>system<|end_header_id|>\\nYou are an expert educational AI tutor. If the context does not contain the answer, you must decline.<|eot_id|><|start_header_id|>user<|end_header_id|>\\nContext Chunks:\\n[1] Newton's second law is defined by the formula F = ma.\\n\\nQuestion: What is the capital city of planet Neptune?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\\n",
        "chosen": "I am sorry, but the provided context does not contain sufficient educational information to answer this question.",
        "rejected": "The capital city of Neptune is Triton City, where methane clouds provide hydroelectric power to ice habitats."
    },
    {
        # Example 3: Precise Citation vs. Missing Citations
        "prompt": "<|start_header_id|>system<|end_header_id|>\\nYou are an expert educational AI tutor. You must provide numeric chunk citations [1].<|eot_id|><|start_header_id|>user<|end_header_id|>\\nContext Chunks:\\n[1] Newton's Third Law states that every action has an equal and opposite reaction.\\n\\nQuestion: What does Newton's Third Law state?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\\n",
        "chosen": "Newton's Third Law states that for every action, there is an equal and opposite reaction [1].",
        "rejected": "Forces are basically equal and opposite whenever things collide."
    }
]

with open("dpo_dataset.jsonl", "w", encoding="utf-8") as f:
    for entry in dpo_data:
        f.write(json.dumps(entry) + "\\n")

print(f"Created dpo_dataset.jsonl with {len(dpo_data)} preference pairs.")"""),

        code_cell("""# Step 5: Configure 4-bit Quantization and Load Base Model
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=True,
)

print(f"Loading Tokenizer for {model_id}...")
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

print(f"Loading 4-bit Quantized Model {model_id}...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=compute_dtype,
    trust_remote_code=True,
)
model = prepare_model_for_kbit_training(model)
model.config.use_cache = False
print("Base model ready for DPO fine-tuning.")"""),

        code_cell("""# Step 6: Configure LoRA for DPO
from peft import LoraConfig, get_peft_model

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()"""),

        code_cell("""# Step 7: Load Preference Dataset
from datasets import load_dataset

dpo_dataset = load_dataset("json", data_files="dpo_dataset.jsonl", split="train")
print(f"Loaded {len(dpo_dataset)} preference pairs.")
print("Sample prompt:", dpo_dataset[0]["prompt"][:120])
print("Sample chosen:", dpo_dataset[0]["chosen"])
print("Sample rejected:", dpo_dataset[0]["rejected"])"""),

        code_cell("""# Step 8: Configure and Run DPOTrainer
from trl import DPOTrainer, DPOConfig

dpo_config = DPOConfig(
    output_dir="./dpo_output",
    beta=0.1,                            # Regularization parameter: higher = stays closer to base policy
    learning_rate=5e-6,                  # DPO uses smaller learning rate than SFT
    lr_scheduler_type="cosine",
    per_device_train_batch_size=1,       # Low batch size to preserve Colab VRAM
    gradient_accumulation_steps=4,
    max_length=512,
    max_prompt_length=256,
    num_train_epochs=3,
    optim="paged_adamw_8bit",
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    logging_steps=1,
    save_strategy="epoch",
    report_to="none",
)

dpo_trainer = DPOTrainer(
    model=model,
    ref_model=None,                      # For PEFT, DPOTrainer disables LoRA adapters for reference policy
    args=dpo_config,
    train_dataset=dpo_dataset,
    tokenizer=tokenizer,
    peft_config=peft_config,
)

print("Starting Direct Preference Optimization (DPO)...")
dpo_trainer.train()
print("DPO Training Completed Successfully!")"""),

        code_cell("""# Step 9: Save DPO-Aligned Adapter
dpo_save_dir = "./educational_dpo_adapter"

dpo_trainer.model.save_pretrained(dpo_save_dir)
tokenizer.save_pretrained(dpo_save_dir)

print(f"DPO-aligned LoRA Adapter saved to '{dpo_save_dir}'!")
!ls -lh {dpo_save_dir}""")
    ]

    return make_notebook(cells)


def main():
    sft_path = PROJECT_ROOT / "train_sft.ipynb"
    dpo_path = PROJECT_ROOT / "train_dpo.ipynb"

    sft_nb = build_sft_notebook()
    with open(sft_path, "w", encoding="utf-8") as f:
        json.dump(sft_nb, f, indent=2)
    print(f"[Generated] {sft_path}")

    dpo_nb = build_dpo_notebook()
    with open(dpo_path, "w", encoding="utf-8") as f:
        json.dump(dpo_nb, f, indent=2)
    print(f"[Generated] {dpo_path}")


if __name__ == "__main__":
    main()
