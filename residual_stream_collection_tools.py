from tl_tools import *
import os
import torch
import yaml
from tqdm import tqdm
from typing import List, Dict, Any
from transformer_lens import HookedTransformer, HookedTransformerConfig
import hashlib


# —————————————————————————————————————————————————————
# 1) Helpers for prompt formatting & file‐naming
# —————————————————————————————————————————————————————


def format_prompt(model, raw_prompt: str, mode: str, remove_bos: bool = True) -> str:
    chat = [{"role": "user", "content": raw_prompt}]
    if mode == "reasoning_boosted":
        chat[0]["content"] = raw_prompt.rstrip() + " Please reason step by step."
    base = model.tokenizer.apply_chat_template(chat,
           add_generation_prompt=True,
           tokenize=False)
    if remove_bos:
        base = base.replace(model.tokenizer.bos_token, "")
    if mode == "immediate_answer":
        base += "</think>\n\n"
    return base


# —————————————————————————————————————————————————————
# 2) Residual‐hook setup & generation
# —————————————————————————————————————————————————————

def collect_resid_post_with_hooks(
    model: HookedTransformer,
    prompt_str: str,
    max_new_tokens: int = 500
):
    n_layers = model.cfg.n_layers
    resid_posts: Dict[int, List[torch.Tensor]] = {l: [] for l in range(n_layers)}

    def make_hook(layer_idx: int):
        def hook_fn(resid, hook):
            # resid: (1, seq_len, d_model) → grab last token resid for this forward
            resid_posts[layer_idx].append(resid[0, -1].detach().cpu())
        return hook_fn

    # 1) Install hooks, run prefix-only forward to capture the prompt's final-token resid
    model.reset_hooks()
    for l in range(n_layers):
        model.add_hook(f"blocks.{l}.hook_resid_post", make_hook(l), dir="fwd")

    input_ids = model.to_tokens(prompt_str)
    prefix_len = input_ids.shape[-1]

    with torch.no_grad():
        _ = model(input_ids, return_type="logits")  # only to invoke hooks once for prefix

    # 2) Re-install hooks afresh so generate() only appends new-token resid
    model.reset_hooks()
    resid_posts = {l: [resid_posts[l][0]] for l in range(n_layers)}  # keep only prefix resid
    for l in range(n_layers):
        model.add_hook(f"blocks.{l}.hook_resid_post", make_hook(l), dir="fwd")

    # 3) Generate new tokens
    with torch.no_grad():
        full_out_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            return_type="tokens"
        )[0].tolist()  # List[int]

    model.reset_hooks()

    # 4) Stack into tensor shape (1 + gen_len, n_layers, d_model)
    layer_tensors = [torch.stack(resid_posts[l], dim=0) for l in range(n_layers)]
    resid_tensor = torch.stack(layer_tensors, dim=1).to(torch.bfloat16)

    return resid_tensor, prefix_len, full_out_ids
    
# —————————————————————————————————————————————————————
# 3) Region‐tagging & distance in token‐space
# —————————————————————————————————————————————————————

def tag_regions_and_distances(model, token_ids: List[int]):
    """
    Two-pass region & distance tagging.
    First pass sets region & dist_from_prev_marker.
    Second pass sets dist_to_next_marker.
    """

    # 1) Get single-token marker IDs
    think_id = model.to_tokens("<think>")[0, -1].item()
    end_id   = model.to_tokens("</think>")[0, -1].item()

    # 2) Prepare outputs
    n = len(token_ids)
    regions     = [None] * n
    dist_from   = [None] * n
    dist_to     = [None] * n

    # 3) Forward pass: last-seen marker → region & dist_from
    last_marker = None  # tuple ("think" or "end", index)
    for i in range(n):
        tok = token_ids[i]
        if tok == think_id:
            last_marker = ("think", i)
        elif tok == end_id:
            last_marker = ("end", i)
        else:
            if last_marker is not None:
                kind, idx0 = last_marker
                regions[i]   = "reasoning" if kind == "think" else "answering"
                dist_from[i] = i - idx0
            # else: leave regions[i], dist_from[i] as None

    # 4) Backward pass: next-seen marker → dist_to
    next_marker_idx = None
    for i in range(n - 1, -1, -1):
        tok = token_ids[i]
        if tok == think_id or tok == end_id:
            next_marker_idx = i
        else:
            if next_marker_idx is not None:
                dist_to[i] = next_marker_idx - i
            # else: leave dist_to[i] as None

    return regions, dist_from, dist_to
    
# —————————————————————————————————————————————————————
# 4) Main loop: generate, tag, save & accumulate metadata
# —————————————————————————————————————————————————————

def run_experiments(
    model: HookedTransformer,
    prompts: List[tuple],
    save_root: str,
    max_new_tokens: int = 750,
    template_modes: List[str] = ["base", "immediate_answer", "reasoning_boosted"]
):
    os.makedirs(save_root, exist_ok=True)

    for subject, raw_prompt in tqdm(prompts, desc="All Prompts"):
        prompt_hash = hash_string(raw_prompt)
        prompt_folder = os.path.join(save_root,
                                     f"{sanitize_subject(subject)}_{prompt_hash}")
        os.makedirs(prompt_folder, exist_ok=True)

        for mode in template_modes:
            resid_path      = os.path.join(prompt_folder, f"{mode}.pt")
            meta_path       = os.path.join(prompt_folder, f"{mode}_metadata.yaml")
            generation_path = os.path.join(prompt_folder, f"{mode}_generation.txt")

            if os.path.exists(meta_path):
                shortened = raw_prompt if len(raw_prompt)<20 else f"{raw_prompt[:20]}..."
                tqdm.write(f"→ Skipping {subject}: {shortened}... [{mode}] (exists)")
                continue

            # 1) Format & collect
            full_prompt = format_prompt(model, raw_prompt, mode)
            resid, prefix_len, token_ids = collect_resid_post_with_hooks(
                model, full_prompt, max_new_tokens
            )

            # 2) Tag full token sequence
            regions, dist_from, dist_to = tag_regions_and_distances(model, token_ids)

            # 3) Save generation text (only new tokens)
            gen_ids = token_ids[prefix_len:]
            generation_txt = model.to_string(gen_ids)
            with open(generation_path, "w", encoding="utf-8") as f:
                f.write(f"{full_prompt}{generation_txt}")

            # 4) Build metadata for each tensor row
            metadata: List[Dict[str,Any]] = []
            total_rows = resid.shape[0]  # = 1 + gen_len
            for ti in range(total_rows):
                tok_pos = (prefix_len - 1) + ti
                token_id = token_ids[tok_pos]
                token_str = model.to_str_tokens(torch.tensor(token_id))[0]
                metadata.append({
                    "tensor_index": ti,
                    "token_index":  tok_pos,
                    "token_id":     token_id,
                    "token_string": token_str,
                    "region":       regions[tok_pos],
                    "dist_from_prev_marker": dist_from[tok_pos],
                    "dist_to_next_marker":   dist_to[tok_pos],
                })

            # 5) Save metadata & tensor
            with open(meta_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(metadata, f, allow_unicode=True)
            torch.save(resid, resid_path)

            # 6) Print the new generation
            sep = "=" * 60
            tqdm.write(f"\n{sep}\nPROMPT [{subject}] MODE={mode}\n{sep}\n{generation_txt}\n")

    print("\n✅ All done — data under", save_root)
