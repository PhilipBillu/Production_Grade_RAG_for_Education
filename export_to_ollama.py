"""
Module: export_to_ollama.py
===========================
Exports Fine-Tuned LoRA / DPO Adapters to Ollama and GGUF Format.

Supports TWO workflows:
1. Workflow A (Full Merge & GGUF Quantization - Recommended):
   - Merges LoRA adapter weights with base 'Meta-Llama-3-8B-Instruct'.
   - Saves merged 16-bit Hugging Face checkpoint.
   - Converts to GGUF format using llama.cpp.
   - Automatically generates the Ollama 'Modelfile'.
   - Registers model in Ollama with `ollama create`.

2. Workflow B (Direct LoRA in Ollama):
   - Shows how to convert adapter weights to GGUF format.
   - Uses the Ollama `ADAPTER` directive on top of `llama3:8b-instruct`.
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def create_modelfile(gguf_path: str, modelfile_path: str = "Modelfile") -> None:
    """Generates an official Ollama Modelfile with Llama-3 template and educational system prompt."""
    content = f"""# Ollama Modelfile for Educational Fine-Tuned Llama-3-8B
FROM {gguf_path}

# Llama-3 Official Chat Template
TEMPLATE \"\"\"{{{{ if .System }}}}<|start_header_id|>system<|end_header_id|>

{{{{ .System }}}}<|eot_id|>{{{{ end }}}}{{{{ if .Prompt }}}}<|start_header_id|>user<|end_header_id|>

{{{{ .Prompt }}}}<|eot_id|>{{{{ end }}}}<|start_header_id|>assistant<|end_header_id|>

{{{{ .Response }}}}<|eot_id|>\"\"\"

# Generation Parameters
PARAMETER temperature 0.2
PARAMETER top_p 0.9
PARAMETER stop "<|eot_id|>"
PARAMETER stop "<|start_header_id|>"

# Educational Assistant System Prompt
SYSTEM \"\"\"You are an expert educational AI assistant. You synthesize factual, well-grounded answers strictly from provided context chunks, with numeric citations [1]. If the context lacks sufficient information, you politely decline to answer.\"\"\"
"""
    with open(modelfile_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"[Ollama] Modelfile created at '{modelfile_path}'.")


def create_direct_adapter_modelfile(adapter_path: str, modelfile_path: str = "Modelfile.adapter") -> None:
    """Creates an Ollama Modelfile that layers a LoRA adapter on top of the base llama3:8b-instruct image."""
    content = f"""# Direct LoRA Adapter Modelfile
FROM llama3:8b-instruct
ADAPTER {adapter_path}

PARAMETER temperature 0.2
PARAMETER top_p 0.9

SYSTEM \"\"\"You are an expert educational AI assistant. You answer strictly using the provided context chunks with citations.\"\"\"
"""
    with open(modelfile_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"[Ollama] Direct adapter Modelfile created at '{modelfile_path}'.")


def merge_lora_to_hf(base_model_id: str, adapter_dir: str, output_dir: str) -> None:
    """Merges LoRA adapter weights with base model and saves full 16-bit model."""
    print(f"\n[Merge] Loading base model: '{base_model_id}'...")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto" if torch.cuda.is_available() else "cpu",
        trust_remote_code=True,
    )

    print(f"[Merge] Loading LoRA adapter from: '{adapter_dir}'...")
    model = PeftModel.from_pretrained(base_model, adapter_dir)

    print("[Merge] Merging LoRA adapter weights into base model (merge_and_unload)...")
    merged_model = model.merge_and_unload()

    print(f"[Merge] Saving merged 16-bit model to: '{output_dir}'...")
    os.makedirs(output_dir, exist_ok=True)
    merged_model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    print(f"[Merge] Merged model successfully saved to '{output_dir}'!")


def convert_to_gguf(merged_hf_dir: str, gguf_output: str, llama_cpp_dir: str = "llama.cpp") -> None:
    """Invokes llama.cpp convert_hf_to_gguf.py to create GGUF model."""
    convert_script = Path(llama_cpp_dir) / "convert_hf_to_gguf.py"

    if not convert_script.exists():
        print(f"\n[Notice] '{convert_script}' not found in current directory.")
        print("To convert the merged model to GGUF, run:")
        print(f"  git clone https://github.com/ggerganov/llama.cpp")
        print(f"  pip install -r llama.cpp/requirements.txt")
        print(f"  python llama.cpp/convert_hf_to_gguf.py {merged_hf_dir} --outfile {gguf_output} --outtype q8_0")
        return

    cmd = [
        sys.executable,
        str(convert_script),
        merged_hf_dir,
        "--outfile",
        gguf_output,
        "--outtype",
        "q8_0",
    ]
    print(f"\n[GGUF] Executing conversion: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"[GGUF] Successfully created '{gguf_output}'!")


def main():
    parser = argparse.ArgumentParser(description="Export Fine-Tuned Llama-3 LoRA to Ollama / GGUF")
    parser.add_argument(
        "--base_model",
        type=str,
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="Base model ID on Hugging Face",
    )
    parser.add_argument(
        "--adapter_dir",
        type=str,
        default="./educational_sft_adapter",
        help="Path to saved LoRA adapter checkpoint (e.g. ./educational_sft_adapter or ./educational_dpo_adapter)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./merged_educational_llama3",
        help="Directory to save merged 16-bit model",
    )
    parser.add_argument(
        "--gguf_output",
        type=str,
        default="./educational_llama3_8b.gguf",
        help="Output path for converted GGUF model",
    )
    parser.add_argument(
        "--skip_merge",
        action="store_true",
        help="Skip merging step if merged model already exists",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("  Educational RAG - LoRA to Ollama / GGUF Export Pipeline")
    print("=" * 70)

    # 1. Merge weights
    if not args.skip_merge:
        if not os.path.exists(args.adapter_dir):
            print(f"[Error] Adapter directory '{args.adapter_dir}' does not exist.")
            print("Please run train_sft.ipynb or train_dpo.ipynb to train your adapter first.")
            print("\nGenerating sample Modelfiles for Ollama reference anyway...")
            create_modelfile(args.gguf_output)
            create_direct_adapter_modelfile(args.adapter_dir)
            return

        merge_lora_to_hf(
            base_model_id=args.base_model,
            adapter_dir=args.adapter_dir,
            output_dir=args.output_dir,
        )
    else:
        print("[Notice] Skipping model merge as requested.")

    # 2. Convert to GGUF
    convert_to_gguf(
        merged_hf_dir=args.output_dir,
        gguf_output=args.gguf_output,
    )

    # 3. Create Ollama Modelfile
    create_modelfile(gguf_path=args.gguf_output, modelfile_path="Modelfile")
    create_direct_adapter_modelfile(adapter_path=args.adapter_dir, modelfile_path="Modelfile.adapter")

    print("\n" + "=" * 70)
    print("  NEXT STEPS TO SERVE IN OLLAMA:")
    print("=" * 70)
    print("1. Build your Ollama model from the Modelfile:")
    print("   ollama create educational-llama3 -f Modelfile")
    print("\n2. Run and chat with your fine-tuned model:")
    print("   ollama run educational-llama3")
    print("\n3. Or point your FastAPI backend to Ollama by setting in .env:")
    print("   OLLAMA_BASE_URL=http://localhost:11434")
    print("   OLLAMA_MODEL=educational-llama3")
    print("=" * 70)


if __name__ == "__main__":
    main()
