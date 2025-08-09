# Core libraries
import json
import numpy as np
import pickle
import torch
from pathlib import Path
from typing import Any, Dict, Union
import datetime
import os

# ML/Analysis libraries
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

# Project-specific imports
from tl_tools import *
from residual_stream_collection_tools import format_prompt

"""
Tools for generating text, analyzing generated text 

This includes
 - extracting OV heads applied to specific resid_pre's
    analyze_head_attention_sources
 - generating text with specific injections 
    simple_patch_generate
    generate_with_resid_post_injection
 - experiments, including 
    -   
        Sweeping over many prompts to evaluate the variance in the ov.\n\n_2_pre
    -
        Sweeping over scales times ov.bos_2_pre and ov.\n\n_2_pre
            N=50, for a subset (eg: 25) prompts 
            For reasoning-formatted prompts, and answering formatted prompts 
    -
        Sweeping over combinations of ov.bos_2_pre and ov.\n\n_2_pre to see which causes 
        answering with the lowest scales 

if __name__ == "__main__", then run experiments 
"""


def analyze_head_attention_sources(model, head, prompt: str, source_idx: int):
    """
    Analyze what a head's OV circuit would contribute from one specific source position.

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
            "vector": contribution,
            "norm": torch.norm(contribution).item(),
            "token": token_strings[source_idx],
            "source_idx": source_idx,
        }

        print("target_token = ", token_strings[source_idx])

        return result

    finally:
        model.reset_hooks()


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
    This is a multi-purpose tool for patching vectors into models and generating
    text

    this can either use a base prompt,
    from which heads are extracted at the final token position.
    Or we can simply inject a supplied vector

    Simplified two-stage head patching:
      1. If base_prompt is provided, run it once to grab each head's final z-output.
         If base_prompt is None, we'll disable those heads (zero them) instead.
      2. Generate on patch_prompt, but at the final prompt token index
         override each target head's z with either the saved vector*scale or zero.
      3. If injection_tensor is given, also add it into attention-out at that same position.

    Args:
        patch_prompt:      The prompt you actually want to generate from.
        target_heads:      One (layer, head) tuple or a list thereof.
        base_prompt:       The prompt to extract "correct" heads from, or None to zero them.
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
        assert (
            model.to_tokens(base_prompt, prepend_bos=False)[0, 0]
            != model.tokenizer.bos_token_id
        ), "Base prompt must not start with BOS token"
    assert (
        model.to_tokens(patch_prompt, prepend_bos=False)[0, 0]
        != model.tokenizer.bos_token_id
    ), "Patch prompt must not start with BOS token"

    # — Stage 1: collect base z's if requested —
    stored_z: Dict[tuple[int, int], torch.Tensor] = {}

    # to confirm that we at most inject once
    injection_count = 0

    if base_prompt is not None:
        # hook each layer's attn.z to grab only the last token
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
                        nonlocal injection_count
                        injection_count += 1
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

    if injection_tensor is not None:
        assert (
            injection_count == 1
        ), f"expected injection_count=1, but {injection_count=}"

    # clean up all hooks
    model.reset_hooks()
    return result


def generate_with_resid_post_injection(
    prompt: str,
    layer_idx: int,
    init_token_tensor: torch.Tensor | None = None,
    body_token_tensor: torch.Tensor | None = None,
    max_new_tokens: int = 50,
    do_sample: bool = False,
    temperature: float | None = None,
) -> str:
    """
    Generate text while injecting into resid_post at `layer_idx`.

    Args:
        prompt:               The text to start from.
        layer_idx:            Which transformer layer to inject into.
        init_token_tensor:    If provided, added once at the final prompt token.
        body_token_tensor:    If provided, added at each generated token step.
        max_new_tokens:       Number of tokens to generate.
        do_sample:            Whether to sample (overrides temperature).
        temperature:          Sampling temperature (if sampling).
    Returns:
        The generated string.
    """
    # --- prepare prompt ---
    tokens = model.to_tokens(prompt, prepend_bos=True)
    prompt_len = tokens.shape[1]
    # make sure user isn't accidentally feeding in a BOS
    assert (
        tokens[0, 0] != model.tokenizer.bos_token_id
    ), "Prompt must not start with BOS token"

    # --- build injection hook ---
    def make_inject():
        def inject_hook(resid_post: torch.Tensor, hook: Any = None) -> torch.Tensor:
            # resid_post: (1, seq_len, d_model)
            seq_len = resid_post.shape[1]
            # initial injection at final prompt token
            if init_token_tensor is not None and seq_len == prompt_len:
                resid_post[0, -1, :] += init_token_tensor.to(resid_post.device)
            # body injection at each new token
            if body_token_tensor is not None and seq_len > prompt_len:
                resid_post[0, -1, :] += body_token_tensor.to(resid_post.device)
            return resid_post

        return inject_hook

    # --- install hook ---
    hook_name = f"blocks.{layer_idx}.hook_resid_post"
    model.add_hook(hook_name, make_inject())

    # --- run generation ---
    result = model.generate(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample or (temperature is not None),
        temperature=temperature,
    )

    # --- cleanup ---
    model.reset_hooks()
    return result


def analyze_ov_outputs(
    prompts: list[str],
    layer_idx: int,
    head: tuple[int, int],
    indices: list[int],
) -> torch.Tensor:
    """
    Analyze OV outputs for a given head across multiple prompts and positions.

    For each prompt, hook the resid_pre at `layer_idx`, capture the activations,
    and apply the OV matrix of `head` to the resid_pre vectors at the specified `indices`.

    Args:
        prompts:     List of input strings to evaluate.
        layer_idx:   Layer index to hook resid_pre from.
        head:        Tuple (layer_idx, head_idx) identifying which head's OV to apply.
        indices:     List of positions (can be negative) within each prompt sequence.

    Returns:
        Tensor of shape (len(indices), len(prompts), d_model) containing OV outputs.
    """
    # Unpack head
    _, head_idx = head

    # OV: (d_model, d_model)
    W_OV = model.OV[layer_idx, head_idx]

    num_indices = len(indices)
    num_prompts = len(prompts)
    d_model = W_OV.shape[0]

    # Prepare output tensor
    outputs = torch.zeros(num_indices, num_prompts, d_model)

    # Hook to capture resid_pre activations
    captured: torch.Tensor | None = None

    def _hook_resid_pre(resid_pre: torch.Tensor, hook=None) -> torch.Tensor:
        nonlocal captured
        captured = resid_pre.detach().clone()
        print("captured shape", captured.shape)
        return resid_pre

    model.add_hook(f"blocks.{layer_idx}.hook_resid_pre", _hook_resid_pre)
    try:
        for i, prompt in enumerate(prompts):
            # Tokenize and run
            tokens = model.to_tokens(prompt, prepend_bos=True)
            captured = None
            with torch.no_grad():
                _ = model.forward(tokens)

            if captured is None:
                raise RuntimeError(f"No resid_pre captured for prompt {i!r}")

            seq_len = captured.shape[1]
            for j, idx in enumerate(indices):
                # handle negative indexing
                pos = idx if idx >= 0 else seq_len + idx
                if not (0 <= pos < seq_len):
                    raise IndexError(
                        f"Index {idx} out of range for prompt length {seq_len}"
                    )

                # extract and project
                resid_vec = captured[0, pos, :]  # (d_model,)
                outputs[j, i, :] = resid_vec @ W_OV  # (d_model,)

    finally:
        model.reset_hooks()

    return outputs.cpu()


class GenerationCollector:
    """
    Collects generation outputs along with associated knob values and lengths,
    and supports saving to and loading from disk.
    """

    def __init__(self, filepath: Union[str, Path]) -> None:
        self.knob_values: list[float] = []  # Fixed: changed back from scale_values
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
            pickle.dump(
                {
                    "knob_values": self.knob_values,
                    "outputs": self.outputs,
                    "lengths": self.lengths,
                },
                f,
            )

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "GenerationCollector":
        """
        Load a collector from disk and return a new instance.
        """
        target = Path(filepath)
        with target.open("rb") as f:
            data = pickle.load(f)

        collector = cls(filepath=target)
        collector.knob_values = data.get("knob_values", [])
        collector.outputs = data.get("outputs", [])
        collector.lengths = data.get("lengths", [])
        return collector


def analyze_prompt_vectors(tensor, null_vector, output_path, verbose=False):
    """
    Analyze variation in prompt vectors and compare to null vector.

    Args:
        tensor: shape (1, n_prompts, d_model) - vectors for each prompt
        null_vector: shape (d_model,) - null/baseline vector
        output_path: str - directory to save results
        verbose: bool - whether to print progress and results to terminal (default: False)
    """
    # Setup
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Convert PyTorch tensors to NumPy arrays
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()
    if isinstance(null_vector, torch.Tensor):
        null_vector = null_vector.detach().cpu().numpy()

    # Extract vectors: (n_prompts, d_model)
    vectors = tensor.squeeze(0)
    n_prompts, d_model = vectors.shape

    if verbose:
        print(f"Analyzing {n_prompts} vectors of dimension {d_model}")

    # 1. Basic statistics
    mean_vector = np.mean(vectors, axis=0)
    component_variances = np.var(vectors, axis=0)
    total_variance = np.sum(component_variances)

    # 2. PCA Analysis
    pca = PCA()
    vectors_pca = pca.fit_transform(vectors)

    # Project null vector into PCA space
    null_pca = pca.transform(null_vector.reshape(1, -1)).flatten()

    # 3. Cosine Similarities
    # All-to-all cosine similarities
    cos_sim_matrix = cosine_similarity(vectors)

    # Cosine similarities to mean
    cos_sim_to_mean = cosine_similarity(vectors, mean_vector.reshape(1, -1)).flatten()

    # Null vector similarities
    null_cos_sim_to_mean = cosine_similarity(
        null_vector.reshape(1, -1), mean_vector.reshape(1, -1)
    )[0, 0]
    null_cos_sims = cosine_similarity(vectors, null_vector.reshape(1, -1)).flatten()

    # 4. Distance Analysis
    # L2 distances from mean
    distances_to_mean = np.linalg.norm(vectors - mean_vector, axis=1)
    null_distance_to_mean = np.linalg.norm(null_vector - mean_vector)

    # Pairwise L2 distances
    pairwise_distances = np.array(
        [
            [np.linalg.norm(vectors[i] - vectors[j]) for j in range(n_prompts)]
            for i in range(n_prompts)
        ]
    )

    # ==================== SAVE NUMERICAL RESULTS ====================

    results = {
        "n_prompts": n_prompts,
        "d_model": d_model,
        "total_variance": total_variance,
        "component_variances": component_variances,
        "pca_explained_variance_ratio": pca.explained_variance_ratio_,
        "pca_cumulative_variance": np.cumsum(pca.explained_variance_ratio_),
        "vectors_pca": vectors_pca,
        "null_pca": null_pca,
        "cos_sim_matrix": cos_sim_matrix,
        "cos_sim_to_mean": cos_sim_to_mean,
        "null_cos_sim_to_mean": null_cos_sim_to_mean,
        "null_cos_sims": null_cos_sims,
        "distances_to_mean": distances_to_mean,
        "null_distance_to_mean": null_distance_to_mean,
        "pairwise_distances": pairwise_distances,
        "mean_vector": mean_vector,
        "null_vector": null_vector,
    }

    np.save(output_path / "analysis_results.npy", results)
    if verbose:
        print(f"Saved numerical results to {output_path / 'analysis_results.npy'}")

    # ==================== CREATE PLOTS ====================

    plt.style.use("default")

    # Figure 1: PCA Analysis
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("PCA Analysis", fontsize=16, fontweight="bold")

    # Explained variance
    axes[0, 0].plot(
        range(1, min(21, len(pca.explained_variance_ratio_) + 1)),
        pca.explained_variance_ratio_[:20],
        "o-",
        linewidth=2,
        markersize=6,
    )
    axes[0, 0].set_xlabel("Principal Component")
    axes[0, 0].set_ylabel("Explained Variance Ratio")
    axes[0, 0].set_title("Explained Variance by Component")
    axes[0, 0].grid(True, alpha=0.3)

    # Cumulative explained variance
    axes[0, 1].plot(
        range(1, min(101, len(pca.explained_variance_ratio_) + 1)),
        np.cumsum(pca.explained_variance_ratio_)[:100],
        "o-",
        linewidth=2,
        markersize=4,
    )
    axes[0, 1].axhline(
        y=0.95, color="red", linestyle="--", alpha=0.7, label="95% variance"
    )
    axes[0, 1].axhline(
        y=0.99, color="orange", linestyle="--", alpha=0.7, label="99% variance"
    )
    axes[0, 1].set_xlabel("Principal Component")
    axes[0, 1].set_ylabel("Cumulative Explained Variance")
    axes[0, 1].set_title("Cumulative Explained Variance")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # PC1 vs PC2 scatter
    axes[1, 0].scatter(
        vectors_pca[:, 0],
        vectors_pca[:, 1],
        alpha=0.7,
        s=60,
        color="steelblue",
        label="Prompt vectors",
    )
    axes[1, 0].scatter(
        null_pca[0],
        null_pca[1],
        color="red",
        s=100,
        marker="x",
        linewidth=3,
        label="Null vector",
    )
    axes[1, 0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    axes[1, 0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    axes[1, 0].set_title("First Two Principal Components")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Intrinsic dimensionality
    dims_for_95 = np.where(np.cumsum(pca.explained_variance_ratio_) >= 0.95)[0]
    dims_for_99 = np.where(np.cumsum(pca.explained_variance_ratio_) >= 0.99)[0]
    dim_95 = (
        dims_for_95[0] + 1
        if len(dims_for_95) > 0
        else len(pca.explained_variance_ratio_)
    )
    dim_99 = (
        dims_for_99[0] + 1
        if len(dims_for_99) > 0
        else len(pca.explained_variance_ratio_)
    )

    axes[1, 1].bar(
        ["Original", "95% variance", "99% variance"],
        [d_model, dim_95, dim_99],
        color=["lightgray", "steelblue", "darkblue"],
        alpha=0.8,
    )
    axes[1, 1].set_ylabel("Effective Dimensions")
    axes[1, 1].set_title("Intrinsic Dimensionality")
    axes[1, 1].set_yscale("log")
    for i, v in enumerate([d_model, dim_95, dim_99]):
        axes[1, 1].text(i, v * 1.1, str(v), ha="center", va="bottom", fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path / "pca_analysis.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Figure 2: Cosine Similarity Analysis
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("Cosine Similarity Analysis", fontsize=16, fontweight="bold")

    # All-to-all cosine similarity heatmap
    mask = np.triu(np.ones_like(cos_sim_matrix), k=1)
    sns.heatmap(
        cos_sim_matrix,
        mask=mask,
        annot=False,
        cmap="RdYlBu_r",
        center=0,
        square=True,
        ax=axes[0, 0],
        cbar_kws={"label": "Cosine Similarity"},
    )
    axes[0, 0].set_title("Pairwise Cosine Similarities")
    axes[0, 0].set_xlabel("Prompt Index")
    axes[0, 0].set_ylabel("Prompt Index")

    # Distribution of cosine similarities (upper triangle only)
    upper_triangle = cos_sim_matrix[np.triu_indices_from(cos_sim_matrix, k=1)]
    axes[0, 1].hist(
        upper_triangle, bins=30, alpha=0.7, color="steelblue", edgecolor="black"
    )
    axes[0, 1].axvline(
        np.mean(upper_triangle),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {np.mean(upper_triangle):.3f}",
    )
    axes[0, 1].set_xlabel("Cosine Similarity")
    axes[0, 1].set_ylabel("Frequency")
    axes[0, 1].set_title("Distribution of Pairwise Cosine Similarities")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Cosine similarities to mean
    axes[1, 0].scatter(
        range(n_prompts), cos_sim_to_mean, alpha=0.7, s=60, color="steelblue"
    )
    axes[1, 0].axhline(
        np.mean(cos_sim_to_mean),
        color="blue",
        linestyle="-",
        alpha=0.7,
        label=f"Mean: {np.mean(cos_sim_to_mean):.3f}",
    )
    axes[1, 0].axhline(
        null_cos_sim_to_mean,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Null: {null_cos_sim_to_mean:.3f}",
    )
    axes[1, 0].set_xlabel("Prompt Index")
    axes[1, 0].set_ylabel("Cosine Similarity to Mean")
    axes[1, 0].set_title("Cosine Similarity to Mean Vector")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Null vector vs others
    axes[1, 1].hist(null_cos_sims, bins=20, alpha=0.7, color="coral", edgecolor="black")
    axes[1, 1].axvline(
        np.mean(null_cos_sims),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {np.mean(null_cos_sims):.3f}",
    )
    axes[1, 1].set_xlabel("Cosine Similarity to Null Vector")
    axes[1, 1].set_ylabel("Frequency")
    axes[1, 1].set_title("Prompt Vectors vs Null Vector")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / "cosine_analysis.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Figure 3: Distance Analysis
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("Distance Analysis", fontsize=16, fontweight="bold")

    # Distances to mean
    axes[0, 0].scatter(
        range(n_prompts), distances_to_mean, alpha=0.7, s=60, color="steelblue"
    )
    axes[0, 0].axhline(
        np.mean(distances_to_mean),
        color="blue",
        linestyle="-",
        alpha=0.7,
        label=f"Mean: {np.mean(distances_to_mean):.2f}",
    )
    axes[0, 0].axhline(
        null_distance_to_mean,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Null: {null_distance_to_mean:.2f}",
    )
    axes[0, 0].set_xlabel("Prompt Index")
    axes[0, 0].set_ylabel("L2 Distance to Mean")
    axes[0, 0].set_title("Distance to Mean Vector")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Distribution of distances to mean
    axes[0, 1].hist(
        distances_to_mean, bins=20, alpha=0.7, color="steelblue", edgecolor="black"
    )
    axes[0, 1].axvline(
        null_distance_to_mean,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Null: {null_distance_to_mean:.2f}",
    )
    axes[0, 1].set_xlabel("L2 Distance to Mean")
    axes[0, 1].set_ylabel("Frequency")
    axes[0, 1].set_title("Distribution of Distances to Mean")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Pairwise distance heatmap
    sns.heatmap(
        pairwise_distances,
        annot=False,
        cmap="viridis",
        square=True,
        ax=axes[1, 0],
        cbar_kws={"label": "L2 Distance"},
    )
    axes[1, 0].set_title("Pairwise L2 Distances")
    axes[1, 0].set_xlabel("Prompt Index")
    axes[1, 0].set_ylabel("Prompt Index")

    # Distribution of pairwise distances
    upper_triangle_dist = pairwise_distances[
        np.triu_indices_from(pairwise_distances, k=1)
    ]
    axes[1, 1].hist(
        upper_triangle_dist, bins=30, alpha=0.7, color="green", edgecolor="black"
    )
    axes[1, 1].axvline(
        np.mean(upper_triangle_dist),
        color="darkgreen",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {np.mean(upper_triangle_dist):.2f}",
    )
    axes[1, 1].set_xlabel("L2 Distance")
    axes[1, 1].set_ylabel("Frequency")
    axes[1, 1].set_title("Distribution of Pairwise Distances")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / "distance_analysis.png", dpi=300, bbox_inches="tight")
    plt.close()

    # ==================== SUMMARY STATISTICS ====================

    # Print summary
    if verbose:
        print("\n" + "=" * 60)
        print("ANALYSIS SUMMARY")
        print("=" * 60)
        print(f"Total variance across prompts: {total_variance:.2f}")
        print(f"Dimensions for 95% variance: {dim_95}/{d_model} ({dim_95/d_model:.1%})")
        print(f"Dimensions for 99% variance: {dim_99}/{d_model} ({dim_99/d_model:.1%})")
        print()
        print("COSINE SIMILARITIES:")
        print(
            f"  Mean pairwise similarity: {np.mean(upper_triangle):.3f} ± {np.std(upper_triangle):.3f}"
        )
        print(
            f"  Mean similarity to centroid: {np.mean(cos_sim_to_mean):.3f} ± {np.std(cos_sim_to_mean):.3f}"
        )
        print(f"  Null similarity to centroid: {null_cos_sim_to_mean:.3f}")
        print(
            f"  Mean null similarity to prompts: {np.mean(null_cos_sims):.3f} ± {np.std(null_cos_sims):.3f}"
        )
        print()
        print("DISTANCES:")
        print(
            f"  Mean distance to centroid: {np.mean(distances_to_mean):.2f} ± {np.std(distances_to_mean):.2f}"
        )
        print(f"  Null distance to centroid: {null_distance_to_mean:.2f}")
        print(
            f"  Mean pairwise distance: {np.mean(upper_triangle_dist):.2f} ± {np.std(upper_triangle_dist):.2f}"
        )

    # Save summary
    summary = {
        "total_variance": total_variance,
        "intrinsic_dim_95": dim_95,
        "intrinsic_dim_99": dim_99,
        "mean_pairwise_cosine": np.mean(upper_triangle),
        "std_pairwise_cosine": np.std(upper_triangle),
        "mean_cosine_to_centroid": np.mean(cos_sim_to_mean),
        "null_cosine_to_centroid": null_cos_sim_to_mean,
        "mean_null_cosine": np.mean(null_cos_sims),
        "mean_distance_to_centroid": np.mean(distances_to_mean),
        "null_distance_to_centroid": null_distance_to_mean,
        "mean_pairwise_distance": np.mean(upper_triangle_dist),
    }

    np.save(output_path / "summary_stats.npy", summary)

    if verbose:
        print(f"\nAll results saved to: {output_path}")
        print(f"Plots: pca_analysis.png, cosine_analysis.png, distance_analysis.png")
        print(f"Data: analysis_results.npy, summary_stats.npy")

    return results, summary


def run_sweep(
    collector,
    injection_vector_data,
    scale_values,
    num_reps,
    prompt,
    target_head=(2, 17),
    max_tokens=1000,
):
    """
    run a single sweep over injection vector scale values
    accumulate within the collector object
    """
    for rep in range(num_reps):
        for injection_scale_value in scale_values:
            out = simple_patch_generate(
                patch_prompt=prompt,
                target_heads=[target_head],
                base_prompt=None,
                max_new_tokens=max_tokens,
                scale=1,
                injection_tensor=(
                    injection_scale_value
                    * injection_vector_data["vector"]
                    / injection_vector_data["vector"].norm(dim=0)
                ),
                temp=0.6,
            )
            length = len(model.to_str_tokens(out))
            collector.add(injection_scale_value, out, length)


def run_variance_experiment(num_prompts, layer_idx, head, variance_exp_dir):
    """
    Extract ov.\n\n_pre vectors from prompts to understand variance.
    Saves results to output folder and performs analysis.
    """
    print(f"Running variance experiment with {num_prompts} prompts...")

    # Get prompts from gsm8k dataset
    _, prompts = zip(*read_prompts(f"gsm8k_{num_prompts}"))

    # Extract outputs from model using analyze_ov_outputs for bulk processing
    outputs = analyze_ov_outputs(
        prompts=prompts, layer_idx=layer_idx, head=head, indices=[-1]
    )
    assert outputs.shape == (1, len(prompts), 4096)

    # Get null prompt baseline (empty prompt) 
    null_output = analyze_ov_outputs(
        prompts=[""], layer_idx=layer_idx, head=head, indices=[-1]
    )
    assert null_output.shape == (1, 1, 4096)

    # Create output directory and save results
    os.makedirs(variance_exp_dir, exist_ok=True)
    torch.save(outputs, f"{variance_exp_dir}/nn_vector_collection.pt")
    torch.save(torch.mean(outputs[0], dim=0), f"{variance_exp_dir}/mean_nn_vector.pt")
    torch.save(null_output[0][0], f"{variance_exp_dir}/null_nn_vector.pt")

    # Save prompts for reference
    with open(f"{variance_exp_dir}/prompts.json", "w", encoding="utf-8") as f:
        json.dump(list(prompts), f, indent=4)

    # Perform analysis
    analyze_prompt_vectors(outputs, null_output[0][0], f"./{variance_exp_dir}/")

    print(f"Variance experiment results saved to: {variance_exp_dir}/")
    return prompts


def prepare_injection_vectors(layer_idx, head, variance_exp_dir):
    """
    Load mean vector from variance experiment and extract BOS vector.
    Returns vector data dicts with same format as analyze_head_attention_sources.
    """
    # Load mean vector from variance experiment
    mean_nn_vector = torch.load(f"{variance_exp_dir}/mean_nn_vector.pt")
    
    # Format as dict like analyze_head_attention_sources returns
    mean_nn_data = {
        "vector": mean_nn_vector,
        "norm": torch.norm(mean_nn_vector).item(),
        "token": "\\n\\n",  # represents end of prompt token
        "source_idx": -1,
    }

    # Extract BOS vector using analyze_head_attention_sources (same as working version)
    bos_data = analyze_head_attention_sources(
        model, head, prompt="", source_idx=0
    )

    return mean_nn_data, bos_data


def run_injection_experiment(
    num_used_prompts,
    layer_idx,
    head,
    num_prompts_injection,
    injection_exp_dir,
    variance_exp_dir,
    max_tokens,
    num_reps,
):
    """
    Run vector injection sweep experiments across different prompt formats,
    vector types, and scale values.
    """
    print("Running injection experiment...")

    os.makedirs(injection_exp_dir, exist_ok=True)

    # Get next set of prompts for injection experiment
    total_prompts_needed = num_used_prompts + num_prompts_injection
    _, all_prompts = zip(*read_prompts(f"gsm8k_{total_prompts_needed}"))
    injection_prompts = all_prompts[-num_prompts_injection:]

    # Save injection prompts for reference
    with open(
        f"{injection_exp_dir}/prompts_for_injection_sweep.json", "w", encoding="utf-8"
    ) as f:
        json.dump(list(injection_prompts), f, indent=4)

    # Prepare vectors for injection (same format as working version)
    mean_nn_data, bos_data = prepare_injection_vectors(
        layer_idx=layer_idx, head=head, variance_exp_dir=variance_exp_dir
    )

    # Experiment configuration
    prompt_formats = ["reason", "immediately_answer"]
    vector_types = [("bos", bos_data), ("nn", mean_nn_data)]
    scale_values = np.linspace(0, 5, 10)
    
    # File-safe timestamp: YYYYMMDD_HHMMSS
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    output_files = []

    # Run sweep across all combinations
    for prompt_idx, prompt in enumerate(injection_prompts):
        for format_type in prompt_formats:
            for vector_name, injection_vector_data in vector_types:

                # Define output file for this combination with timestamp
                output_file = (
                    f"{injection_exp_dir}/{format_type}_prompt_{prompt_idx}_"
                    f"{vector_name}_vector_sweep_{timestamp}.pkl"
                )
                output_files.append(output_file)

                # Set up data collector
                collector = GenerationCollector(filepath=output_file)
                formatted_prompt = format_prompt(model, prompt, format_type)

                # Run the sweep (fixed to use same normalization as working version)
                run_sweep(
                    collector=collector,
                    injection_vector_data=injection_vector_data,
                    scale_values=scale_values,
                    num_reps=num_reps,
                    prompt=formatted_prompt,
                    max_tokens=max_tokens,
                )

    print("Injection experiment results saved to:")
    for file_path in output_files:
        print(f"  {file_path}")

    return output_files


if __name__ == "__main__":
    # Constants
    LAYER_IDX = 2
    HEAD = (2, 17)
    NUM_PROMPTS_VARIANCE = 100
    NUM_PROMPTS_INJECTION = 20
    VARIANCE_EXP_DIR = "nn_vector_variance_experiment"
    INJECTION_EXP_DIR = "vector_injection_and_generation_experiment"
    MAX_TOKENS = 1000
    NUM_INJECTION_REPS = 50

    torch.set_grad_enabled(False)

    # Load model
    model = load_llama8br1()

    # Run variance analysis experiment
    variance_prompts = run_variance_experiment(
        num_prompts=NUM_PROMPTS_VARIANCE,
        layer_idx=LAYER_IDX,
        head=HEAD,
        variance_exp_dir=VARIANCE_EXP_DIR,
    )

    # Run vector injection experiment
    injection_results = run_injection_experiment(
        num_used_prompts=len(variance_prompts),
        num_prompts_injection=NUM_PROMPTS_INJECTION,
        layer_idx=LAYER_IDX,
        head=HEAD,
        injection_exp_dir=INJECTION_EXP_DIR,
        variance_exp_dir=VARIANCE_EXP_DIR,
        max_tokens=MAX_TOKENS,
        num_reps=NUM_INJECTION_REPS,
    )

    print(f"\nExperiment complete. Results saved to:")
    print(f"  Variance analysis: {VARIANCE_EXP_DIR}/")
    print(f"  Injection sweeps: {INJECTION_EXP_DIR}/")