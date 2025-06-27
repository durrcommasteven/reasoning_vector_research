from tl_tools import *
from residual_stream_collection_tools import format_prompt

torch.set_grad_enabled(False)

model = load_llama8br1()

def analyze_head_attention_sources(model, head, prompt: str, source_idx: int):
    """
    Analyze what this head's OV circuit would contribute from one specific source position.
    
    Args:
        model: HookedTransformer model
        head: (layer_idx, head_idx) tuple  
        prompt: Input prompt
        source_idx: The source position to analyze (can be negative for indexing from end)
        
    Returns:
        dict: Single contribution result
    """
    
    layer_idx, head_idx = head
    captured_resid_pre = None
    
    def hook_resid_pre(activation, hook):
        nonlocal captured_resid_pre
        captured_resid_pre = activation.detach().clone()
        return activation
    
    model.add_hook(f"blocks.{layer_idx}.hook_resid_pre", hook_resid_pre)
    
    try:
        # Tokenize with explicit BOS control
        tokens = model.to_tokens(prompt, prepend_bos=True)
        
        with torch.no_grad():
            _ = model.forward(tokens)
        
        # Handle negative indexing
        seq_len = captured_resid_pre.shape[1]
        if source_idx < 0:
            source_idx = seq_len + source_idx
            
        # Get OV matrix directly from TransformerLens
        W_OV = model.OV[layer_idx, head_idx]  # (d_model, d_model)
        
        # Compute contribution from the specific source position
        source_resid = captured_resid_pre[0, source_idx, :]
        contribution = source_resid @ W_OV
        
        # Get token string for this position  
        token_strings = model.to_str_tokens(tokens[0])  # Use the same tokenized input
        
        result = {
            'vector': contribution,
            'norm': torch.norm(contribution).item(),
            'token': token_strings[source_idx],
            'source_idx': source_idx
        }

        print("target_token = ",token_strings[source_idx])
        
        return result
        
    finally:
        model.reset_hooks()

import torch

def simple_patch_generate(
    patch_prompt: str,
    target_heads: list[tuple[int, int]] | tuple[int, int],
    base_prompt: str | None = None,
    max_new_tokens: int = 50,
    scale: float = 1.0,
    injection_tensor: torch.Tensor | None = None,
    temp: float | None = None,
) -> str:
    """
    Simplified two-stage head patching:
      1. If base_prompt is provided, run it once to grab each head's final z-output.
         If base_prompt is None, we’ll disable those heads (zero them) instead.
      2. Generate on patch_prompt, but at the final prompt token index
         override each target head’s z with either the saved vector*scale or zero.
      3. If injection_tensor is given, also add it into attention-out at that same position.

    Args:
        patch_prompt:      The prompt you actually want to generate from.
        target_heads:      One (layer, head) tuple or a list thereof.
        base_prompt:       The prompt to extract “correct” heads from, or None to zero them.
        max_new_tokens:    How many new tokens to sample after patching.
        scale:             Multiplicative factor on saved head activations.
        injection_tensor:  A (d_model,) tensor to add into attn.out at the patch position.
    Returns:
        The decoded string from the patched generation.
    """
    # — normalize heads —
    if isinstance(target_heads, tuple):
        heads: list[tuple[int, int]] = [target_heads]
    else:
        heads = target_heads
    
    # for each head, assert that the first token is not a bos token
    if base_prompt is not None:
        assert model.to_tokens(base_prompt, prepend_bos=False)[0, 0] != model.tokenizer.bos_token_id, "Base prompt must not start with BOS token"
    assert model.to_tokens(patch_prompt, prepend_bos=False)[0, 0] != model.tokenizer.bos_token_id, "Patch prompt must not start with BOS token"

    # — Stage 1: collect base z’s if requested —
    stored_z: Dict[tuple[int, int], torch.Tensor] = {}

    if base_prompt is not None:
        # hook each layer’s attn.z to grab only the last token
        for layer_idx, head_idx in heads:
            hook_name = f"blocks.{layer_idx}.attn.hook_z"
            def make_save(layer_idx: int, head_idx: int):
                def save_hook(z: torch.Tensor, hook=None) -> torch.Tensor:
                    # z: (1, seq_len, n_heads, head_dim)
                    stored_z[(layer_idx, head_idx)] = (
                        z[0, -1, head_idx, :].detach().cpu().clone()
                    )
                    return z
                return save_hook
            fn = make_save(layer_idx, head_idx)
            model.add_hook(hook_name, fn)

        # run base_prompt through the model (no new tokens needed)
        tokens = model.to_tokens(base_prompt, prepend_bos=True)
        
        # collect the final z values
        with torch.no_grad():
            _ = model.forward(tokens)
        
        # we've collected the z values, so we can remove the hooks
        model.reset_hooks()

    # — Stage 2: install patch hooks for z —
    # precompute prompt length
    tokens_patch = model.to_tokens(patch_prompt, prepend_bos=True)
    prompt_len = tokens_patch.shape[1]
    for layer_idx, head_idx in heads:
        hook_name = f"blocks.{layer_idx}.attn.hook_z"
        def make_patch(layer_idx: int, head_idx: int):
            def patch_hook(z: torch.Tensor, hook=None) -> torch.Tensor:
                # only override at the final prompt index
                # z: (1, seq_len, n_heads, head_dim)
                if z.shape[1] == prompt_len:
                    if base_prompt is None:
                        z[0, -1, head_idx, :] *= 0.0
                    else:
                        vec = stored_z[(layer_idx, head_idx)].to(z.device) * scale
                        z[0, -1, head_idx, :] = vec
                return z
            return patch_hook
        
        fn = make_patch(layer_idx, head_idx)
        model.add_hook(hook_name, fn)

    # — optional: injection into attention output —
    if injection_tensor is not None:
        for layer_idx, _ in heads:
            hook_name = f"blocks.{layer_idx}.hook_resid_mid"
            def make_inject():
                def inject_hook(out: torch.Tensor, hook=None) -> torch.Tensor:
                    # out: (1, seq_len, d_model)
                    if out.shape[1] == prompt_len:
                        out[0, -1, :] += injection_tensor.to(out.device)
                    return out
                return inject_hook
            fn = make_inject()
            model.add_hook(hook_name, fn)

    # — run patched generation —
    result = model.generate(
        patch_prompt,
        max_new_tokens=max_new_tokens,
        do_sample=False if temp is None else True,
        temperature=temp,
    )

    # clean up all hooks
    model.reset_hooks()
    return result


reason_prompt = format_prompt(model, "Whats the fifth prime?", "reason")

immediate_answer_prompt = format_prompt(model, "Whats the fifth prime?", "immediate_answer")

print(reason_prompt)
print(immediate_answer_prompt)


bos_values = analyze_head_attention_sources(
    model, 
    (2, 17), 
    prompt = format_prompt(model, "Whats the fifth prime?", "immediate_answer"),
    source_idx = 0
)

final_nn_values = analyze_head_attention_sources(
    model, 
    (2, 17), 
    prompt = format_prompt(model, "Whats the fifth prime?", "immediate_answer"),
    source_idx = -1
)



import numpy as np
import pickle
from pathlib import Path
from typing import Any, Union


class GenerationCollector:
    """
    Collects generation outputs along with associated knob values and lengths,
    and supports saving to and loading from disk.
    """

    def __init__(self, filepath: Union[str, Path]) -> None:
        self.knob_values: list[float] = []
        self.outputs: list[Any] = []
        self.lengths: list[int] = []
        self.filepath: Path = Path(filepath)
        # Ensure directory exists
        if not self.filepath.parent.exists():
            self.filepath.parent.mkdir(parents=True, exist_ok=True)

    def add(self, knob_value: float, output: Any, length: int) -> None:
        """
        Add a new generation result and save to disk.
        """
        self.knob_values.append(knob_value)
        self.outputs.append(output)
        self.lengths.append(length)
        self.save()

    def save(self, filepath: Union[str, Path] | None = None) -> None:
        """
        Save the collector data to disk via pickle.
        """
        target = Path(filepath) if filepath is not None else self.filepath
        with target.open("wb") as f:
            pickle.dump({
                'knob_values': self.knob_values,
                'outputs': self.outputs,
                'lengths': self.lengths
            }, f)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "GenerationCollector":
        """
        Load a collector from disk and return a new instance.
        """
        target = Path(filepath)
        with target.open("rb") as f:
            data = pickle.load(f)

        collector = cls(filepath=target)
        collector.knob_values = data.get('knob_values', [])
        collector.outputs = data.get('outputs', [])
        collector.lengths = data.get('lengths', [])
        return collector


# a string, file-safe version of the current date
import datetime
date_str = datetime.datetime.now().strftime("%Y%m%d")

# Define sweep and output collector
reason_knob_values = np.linspace(0, 5, 10)
output_file = f"reason_prompt_reason_vector_sweep_{date_str}.pkl"
collector = GenerationCollector(filepath=output_file)

# Run sweep
for rep in range(50):
    for knob_value in reason_knob_values:
        out = simple_patch_generate(
            patch_prompt=reason_prompt,
            target_heads=[(2, 17)],
            base_prompt=None,
            max_new_tokens=1000,
            scale=1,
            injection_tensor=(knob_value * bos_values['vector'] / bos_values['vector'].norm(dim=0)),
            temp=0.6,
        )
        length = len(model.to_str_tokens(out))
        collector.add(knob_value, out, length)

print(f"Results saved to {output_file}")


reason_knob_values = np.linspace(0, 5, 10)
output_file = f"reason_prompt_answer_vector_sweep_{date_str}.pkl"
collector = GenerationCollector(filepath=output_file)

# Run sweep using nested rep then knob_value loops
for rep in range(50):
    for knob_value in reason_knob_values:
        current_out = simple_patch_generate(
            patch_prompt=reason_prompt,
            target_heads=[(2, 17)],
            base_prompt=None,
            max_new_tokens=1000,
            scale=1.0,
            injection_tensor = knob_value*final_nn_values['vector']/final_nn_values['vector'].norm(dim=0),
            temp=0.6,
        )
        length = len(model.to_str_tokens(current_out))
        collector.add(knob_value, current_out, length)

print(f"Results saved to {output_file}")


