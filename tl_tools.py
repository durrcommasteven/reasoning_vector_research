# Core ML Libraries
import torch
import torch.nn.functional as F
import numpy as np

# Transformer Lens
from transformer_lens import HookedTransformer
import transformer_lens
import transformer_lens.utils as utils

# Hugging Face Transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

# Utilities
import os
import hashlib
import yaml
import pickle
from typing import Dict, List, Tuple, Callable

# Plotting & Progress
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

import re
from typing import List, Tuple, Optional

def load_llama8br1(local_cache_dir: str = "./hf_cache") -> HookedTransformer:
    torch.set_grad_enabled(False)
    device = utils.get_device()

    reference_model_path = "meta-llama/Llama-3.1-8B"
    baseline_model_path = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    local_baseline_path = os.path.join(local_cache_dir, "deepseek-8b")

    # Try to load from local path
    try:
        baseline_model_hf = AutoModelForCausalLM.from_pretrained(
            baseline_model_path,
            torch_dtype=torch.bfloat16,
            cache_dir=local_baseline_path,
            local_files_only=True,  # <– avoids internet
        )
        baseline_model_tokenizer = AutoTokenizer.from_pretrained(
            baseline_model_path,
            cache_dir=local_baseline_path,
            local_files_only=True,
        )
        print("Loaded model and tokenizer from local cache.")
    except Exception as e:
        print(f"Local load failed: {e}")
        print("Falling back to online download...")
        baseline_model_hf = AutoModelForCausalLM.from_pretrained(
            baseline_model_path,
            torch_dtype=torch.bfloat16,
            cache_dir=local_baseline_path,
            local_files_only=False,
        )
        baseline_model_tokenizer = AutoTokenizer.from_pretrained(
            baseline_model_path,
            cache_dir=local_baseline_path,
            local_files_only=False,
        )

    model = HookedTransformer.from_pretrained_no_processing(
        reference_model_path,
        hf_model=baseline_model_hf,
        tokenizer=baseline_model_tokenizer,
        device=device,
        move_to_device=True,
    )

    return model

"""
def load_llama8br1():
    torch.set_grad_enabled(False)

    device = utils.get_device()

    reference_model_path = "meta-llama/Llama-3.1-8B"
    baseline_model_path = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

    baseline_model_hf = AutoModelForCausalLM.from_pretrained(
        baseline_model_path, torch_dtype=torch.bfloat16
    )
    baseline_model_tokenizer = AutoTokenizer.from_pretrained(baseline_model_path)

    model = HookedTransformer.from_pretrained_no_processing(
        reference_model_path,
        hf_model=baseline_model_hf,
        tokenizer=baseline_model_tokenizer,
        device=device,
        move_to_device=True,
    )

    return model
"""

def extract_num_prompts(s: str) -> int:
    try:
        return int(s.split('_')[-1])
    except (ValueError, IndexError):
        raise ValueError(f"String '{s}' is not in the correct format (e.g., 'prefix_123')")


def read_prompts(
    output_type: Optional[str] = None, path_to_prompts="mds/reasoning-prompts.md"
) -> List[Tuple[str, str]] | List[Tuple[Optional[str], str]]:
    """
    Reads reasoning prompts from a Markdown file.

    Args:
        output_type: If a string matching a level ## heading in the file,
                     only prompts under that heading are returned with the
                     heading as the output type. If None, all prompts are
                     returned with their respective heading as the output type.
        path_to_prompts: The path to the Markdown file containing the prompts.

    Returns:
        A list of tuples. Each tuple contains the output type (the heading or None)
        and the corresponding prompt.
    """
    if 'gsm8k' in output_type:
        # take from gsm8k 
        from datasets import load_dataset
        ds = load_dataset("openai/gsm8k", "main")

        if output_type != 'gsm8k':
            # expect a number as well to indicate the number of prompts to select
            num_prompts = extract_num_prompts(output_type)
        else:
            num_prompts = len(ds['train']['question'])
        
        return [(output_type, prompt) for prompt in ds['train']['question'][:num_prompts]]
    
    try:
        with open(path_to_prompts, "r") as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Error: File not found at {path_to_prompts}")
        return []

    prompts = []
    current_heading = None
    lines = content.splitlines()

    for line in lines:
        heading_match = re.match(r"##\s+(.+)", line)
        prompt_match = re.match(r"^\d+\.\s+(.+)", line)

        if heading_match:
            current_heading = heading_match.group(1).replace(" Prompts", "").strip()
        elif prompt_match:
            prompt_text = prompt_match.group(1).strip()
            if output_type is None:
                prompts.append((current_heading.lower(), prompt_text))
            elif current_heading.lower() == output_type.lower():
                prompts.append((output_type.lower(), prompt_text))

    return prompts


def hash_string(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:10]


def sanitize_subject(s: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in s).lower().strip("_")


#def make_file_name(prompt_subject, prompt_string, template_mode):
#    clean = sanitize_filename(prompt_subject + "_" + prompt_string)
#    h = hash_string(prompt_string + template_mode)
#    return f"{clean}_{template_mode}_{h}"


def save_pickle(obj, filename):
    """
    Save an object to a pickle file.

    Args:
        obj: The object to save.
        filename (str): The name of the file to save the object to.
    """
    try:
        with open(filename, "wb") as f:
            pickle.dump(obj, f)
    except Exception as e:
        raise IOError(f"Failed to save pickle to {filename}: {e}")


def load_pickle(filename):
    """
    Load an object from a pickle file.

    Args:
        filename (str): The name of the file to load the object from.

    Returns:
        The loaded object.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"No such file: {filename}")
    try:
        with open(filename, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        raise IOError(f"Failed to load pickle from {filename}: {e}")


def js_distance(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Jensen-Shannon distance between p and q.
    Returns sqrt(0.5*KL(p||m) + 0.5*KL(q||m)), where m = 0.5*(p+q).
    """
    m = 0.5 * (p + q)
    kl_pm = torch.sum(p * (torch.log(p + eps) - torch.log(m + eps)))
    kl_qm = torch.sum(q * (torch.log(q + eps) - torch.log(m + eps)))
    jsd = 0.5 * (kl_pm + kl_qm)
    return torch.sqrt(jsd)