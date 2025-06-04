import os
import torch
import yaml
import numpy as np
from typing import Union, List, Dict, Any, Set, Optional, Tuple
import time
from datetime import timedelta
from tqdm import tqdm

# -----------------------------------------
# Configuration of comparisons
# -----------------------------------------
COMPARISONS = [
    {
        "name": "base_answer_vs_enhanced_reason_body",
        "mode1": "base",
        "region1": "answering",
        "mode2": "reasoning_boosted",
        "region2": "reasoning",
        "subset": "body",
        "normalize": True
    },
    {
        "name": "base_reason_vs_base_answer_body",
        "mode1": "base",
        "region1": "reasoning",
        "mode2": "base",
        "region2": "answering",
        "subset": "body",
        "normalize": True
    },
    {
        "name": "enhanced_reason_vs_enhanced_answer_body",
        "mode1": "reasoning_boosted",
        "region1": "reasoning",
        "mode2": "reasoning_boosted",
        "region2": "answering",
        "subset": "body",
        "normalize": True
    },
    {
        "name": "base_reason_vs_immediate_answer_body",
        "mode1": "base",
        "region1": "reasoning",
        "mode2": "immediate_answer",
        "region2": "answering",
        "subset": "body",
        "normalize": True
    },
    {
        "name": "enhanced_reason_vs_base_reason_body",
        "mode1": "reasoning_boosted",
        "region1": "reasoning",
        "mode2": "base",
        "region2": "reasoning",
        "subset": "body",
        "normalize": True
    },
    {
        "name": "base_reason_vs_base_answer_initial",
        "mode1": "base",
        "region1": "reasoning",
        "mode2": "base",
        "region2": "answering",
        "subset": "initial",
        "normalize": False
    },
    {
        "name": "enhanced_reason_vs_enhanced_answer_initial",
        "mode1": "reasoning_boosted",
        "region1": "reasoning",
        "mode2": "reasoning_boosted",
        "region2": "answering",
        "subset": "initial",
        "normalize": False
    },
    {
        "name": "enhanced_reason_vs_base_reason_initial",
        "mode1": "reasoning_boosted",
        "region1": "reasoning",
        "mode2": "base",
        "region2": "reasoning",
        "subset": "initial",
        "normalize": False
    },
    {
        "name": "base_reason_vs_immediate_answer_initial",
        "mode1": "base",
        "region1": "reasoning",
        "mode2": "immediate_answer",
        "region2": "answering",
        "subset": "initial",
        "normalize": False
    }
]
# -----------------------------------------
# Utility Functions
# -----------------------------------------

def load_tensor(path: str) -> Optional[np.ndarray]:
    """Load bf16 tensor as float32 numpy, return None on failure."""
    try:
        return torch.load(path).float().cpu().numpy()
    except Exception as e:
        print(f"Error loading tensor from {path}: {e}")
        return None


def load_metadata(path: str) -> Optional[List[Dict[str, Any]]]:
    """Load YAML metadata, return None on failure."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading metadata from {path}: {e}")
        return None


def lda_per_layer_with_proj(
    data_1: dict[int, np.ndarray] | np.ndarray,
    data_2: dict[int, np.ndarray] | np.ndarray,
    reg: float = 0.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Compute two‐class LDA directions per layer ℓ, plus projections of each sample
    onto its layer‐specific LDA vector.

    Args:
        Either 
            dict1: mapping → np.ndarray of shape (B_i, L, D).  All arrays share the same L, D.
            dict2: same type of mapping for class 2.
        Or
            a np.ndarray of shape (B_total, L, D). 
        reg (float): optional ridge penalty added to each within‐class scatter S_w[ℓ].

    Returns:
        w:     np.ndarray of shape (L, D),  where w[ℓ] ∝ (S_w[ℓ])⁻¹·(μ1[ℓ] − μ2[ℓ]).
        proj1: np.ndarray of shape (N1, L, 1), projections of class‐1 samples.
        proj2: np.ndarray of shape (N2, L, 1), projections of class‐2 samples.
        (resids_1, resids_2): Tuple[np.ndarray, np.ndarray], the residuals of the data
            (these are returned for use in computing the proj_mean_diff)

    Internals summary:
      1) Concatenate all (B_i, L, D) from dict1 → X1 of shape (N1, L, D).
         Concatenate all from dict2 → X2 of shape (N2, L, D).
      2) Compute per‐layer means μ1[ℓ,:], μ2[ℓ,:] in ℝ^D.
      3) Center X1,X2 → X1c, X2c.
      4) Build within‐class scatter S_w[ℓ] = Σ (x − μ)(x − μ)^T for both classes.
      5) Solve S_w[ℓ] · w[ℓ] = μ1[ℓ] − μ2[ℓ]  for each ℓ → w[ℓ] ∈ ℝ^D.
      6) Project each sample:  proj1[n, ℓ, 0] = X1[n,ℓ,:] · w[ℓ], etc.
    """
    if isinstance(data_1, dict):
        assert isinstance(data_2, dict)
        # 1) Concatenate all batches in dict1 → X1 (N1×L×D), and dict2 → X2 (N2×L×D)
        X1 = np.concatenate(list(data_1.values()), axis=0)  # (N1, L, D)
        X2 = np.concatenate(list(data_2.values()), axis=0)  # (N2, L, D)
    elif isinstance(data_1, np.ndarray):
        assert isinstance(data_2, np.ndarray)
        # 1) Concatenate all batches in list1 → X1 (N1×L×D), and list2 → X2 (N2×L×D)
        N1, L1, D1 = data_1.shape
        X1 = data_1
        N2, L2, D2 = data_2.shape
        X2 = data_2
        if L1 != L2 or D1 != D2:
            raise ValueError(f"Shape mismatch: data_1 has {L1, D1}, data_2 has {L2, D2}")
    else:
        raise ValueError(f"Invalid data type: {type(data_1)}")

    # Check that L and D match
    if X1.shape[1:] != X2.shape[1:]:
        raise ValueError(f"Shape mismatch: dict1 has {X1.shape[1:]}, dict2 has {X2.shape[1:]}")

    N1, L, D = X1.shape
    N2 = X2.shape[0]

    # 2) Per‐layer means
    mu1 = X1.mean(axis=0)  # (L, D)
    mu2 = X2.mean(axis=0)  # (L, D)

    # 3) Center the data
    X1c = X1 - mu1[None, :, :]  # (N1, L, D)
    X2c = X2 - mu2[None, :, :]  # (N2, L, D)

    # 4) Within‐class scatter via einsum
    #   S1[ℓ] = Σ_n (X1c[n,ℓ,:] @ X1c[n,ℓ,:]^T)  → shape (L, D, D)
    #   S2[ℓ] = Σ_m (X2c[m,ℓ,:] @ X2c[m,ℓ,:]^T)  → shape (L, D, D)
    S1 = np.einsum('nld,nle->lde', X1c, X1c)  # (L, D, D)
    S2 = np.einsum('nld,nle->lde', X2c, X2c)  # (L, D, D)

    S_w = S1 + S2
    if reg > 0.0:
        I = np.eye(D, dtype=S_w.dtype)[None, :, :]  # (1, D, D)
        S_w = S_w + reg * I

    # 5) Delta means per layer: Δμ[ℓ] = μ1[ℓ] − μ2[ℓ]  → shape (L, D)
    delta_mu = mu1 - mu2  # (L, D)

    # Allocate w
    w = np.zeros((L, D), dtype=X1.dtype)

    # Solve S_w[ℓ] · w[ℓ] = Δμ[ℓ] for each ℓ
    for ℓ in range(L):
        w[ℓ] = np.linalg.solve(S_w[ℓ], delta_mu[ℓ])

    # 6) Compute projections along the direction w[ℓ]:
    #    proj1[n,ℓ] = X1[n,ℓ,:] · w[ℓ],  for n=0..N1−1
    #    proj2[m,ℓ] = X2[m,ℓ,:] · w[ℓ],  for m=0..N2−1
    #
    # We can do this with einsum, yielding shape (N1, L) and (N2, L)
    proj1_flat = np.einsum('nld,ld->nl', X1, w)  # (N1, L)
    proj2_flat = np.einsum('nld,ld->nl', X2, w)  # (N2, L)

    # Finally, reshape to (N1, L, 1) and (N2, L, 1)
    proj1 = proj1_flat[:, :, None]  # (N1, L, 1)
    proj2 = proj2_flat[:, :, None]  # (N2, L, 1)

    return w, proj1, proj2, (X1, X2)


def filter_indices(metadata: List[Dict[str, Any]], region: str, subset: str) -> Tuple[List[int], List[int]]:
    """
    Filter by region, subset, and token_id <128000.
    return a list of idxs and a list of respective tokens 
    """
    idxs = []
    tokens = []
    for i, m in enumerate(metadata):
        if m.get('region') != region: continue
        tok_id = m.get('token_id', -1)
        if tok_id < 0 or tok_id >= 128000: continue
        dist = m.get('dist_from_prev_marker')
        if subset == 'body' and dist is not None and dist > 1:
            idxs.append(i)
            tokens.append(tok_id)
        elif subset == 'initial' and dist == 1:
            idxs.append(i)
            tokens.append(tok_id)
    return idxs, tokens


def collect_token_data(
        data_root: str,
        mode1: str, region1: str,
        mode2: str, region2: str, 
        subject: None | str | list[str] = None,
        restrict_to_common_tokens: bool = False,
) -> Dict[str, Dict[int, Dict[str, List[int]]]]:
    """
    Collect token_ids that appear in both mode1/region1 and mode2/region2 across all prompts.
    return two dictionaries mapping token_id to a list of (folder, list of tensor_indices)

    if restrict_to_common_tokens:
        then we'll only collect tokens that appear in both mode1/region1 and mode2/region2
        This is useful when doing token normalized comparisons
    
    if subject is None:
        then we'll collect all tokens from all subjects
    elif isinstance(subject, str):
        then we'll collect all tokens from the subject
    elif isinstance(subject, list):
        then we'll collect all tokens from the subjects in the list
    """
    # collect the tokens appearing in each subset
    # map these to their respective files and indices
    # eg: token_to_idxs1[123] = [(file1, [1, 2, 3]), (file2, [4, 5, 6])]
    token_dict_1, token_dict_2 = dict(), dict() 
    # collect all relevant folders
    if subject is None:
        data_folders = os.listdir(data_root)
    elif isinstance(subject, str):
        data_folders = [f for f in os.listdir(data_root) if f.startswith(subject.lower())]
    elif isinstance(subject, list):
        data_folders = [f for f in os.listdir(data_root) if f.split('_')[0] in subject]
    else:
        raise ValueError(f"Invalid subject type: {type(subject)}")

    # turn these into folder_paths
    data_folders = sorted([
        os.path.join(data_root, f) for f in data_folders if os.path.isdir(os.path.join(data_root, f))
    ])

    # collect the tokens appearing in each subset
    for folder in data_folders:
        meta1 = load_metadata(os.path.join(folder, f"{mode1}_metadata.yaml")) or []
        meta2 = load_metadata(os.path.join(folder, f"{mode2}_metadata.yaml")) or []
        for m in meta1:
            if m.get('region')==region1 and 0<=m.get('token_id',-1)<128000:
                # add it to the dict, noting that each folder may have multiple tensor_indices
                # with the same token_id
                if m['token_id'] not in token_dict_1:
                    token_dict_1[m['token_id']] = {folder: [m['tensor_index']]}
                elif folder not in token_dict_1[m['token_id']]:
                    token_dict_1[m['token_id']].update({
                        folder: [m['tensor_index']]
                    })
                else:
                    token_dict_1[m['token_id']][folder].append(m['tensor_index'])
        for m in meta2:
            if m.get('region')==region2 and 0<=m.get('token_id',-1)<128000:
                if m['token_id'] not in token_dict_2:
                    token_dict_2[m['token_id']] = {folder: [m['tensor_index']]}
                elif folder not in token_dict_2[m['token_id']]:
                    token_dict_2[m['token_id']].update({
                        folder: [m['tensor_index']]
                    })
                else:
                    token_dict_2[m['token_id']][folder].append(m['tensor_index'])
    
    if restrict_to_common_tokens:
        # only keep tokens that appear in both mode1/region1 and mode2/region2
        token_dict_1 = {k:v for k, v in token_dict_1.items() if k in token_dict_2}
        token_dict_2 = {k:v for k, v in token_dict_2.items() if k in token_dict_1}
    
    return {
        'token_dict_1': token_dict_1,
        'token_dict_2': token_dict_2
    }


def compute_token_normalized_diffs(
        data_root: str,
        comparison: dict[str, str],
        layer_idxs: List[int],
        subject: None | str | list[str] = None,
) -> Dict[str, np.ndarray]:
    """
    given a comparison dict, and a data_root, compute the token normalized diffs

    first extract token_dict_1 and token_dict_2 (restricting to common tokens)
    each is a dict mapping token_id to a list of (folder, list of tensor_indices)

    Normalize the tokens by subtracting the mean of the tokens in both subsets
    return the mean difference and the lda direction for each layer
    """

    # note we restrict to common tokens here since we're doing token normalized comparisons
    token_data_dicts = collect_token_data(
        data_root=data_root,
        mode1=comparison['mode1'], region1=comparison['region1'],
        mode2=comparison['mode2'], region2=comparison['region2'], 
        subject=subject,
        restrict_to_common_tokens=True,
    )

    common_token_dict_1, common_token_dict_2 = token_data_dicts['token_dict_1'], token_data_dicts['token_dict_2']

    assert len(common_token_dict_1) == len(common_token_dict_2)

    # collect a dict mapping folder to a dict of {token_id: indices}
    # so that we may easily go folder by folder (loading takes time)
    folder_to_token_id_1 = dict()
    folder_to_token_id_2 = dict()
    for tok_id, folders_indices in common_token_dict_1.items():
        for folder, indices in folders_indices.items():
            folder_to_token_id_1.setdefault(folder, {}).update({tok_id: indices})
    for tok_id, folders_indices in common_token_dict_2.items():  
        for folder, indices in folders_indices.items():
            folder_to_token_id_2.setdefault(folder, {}).update({tok_id: indices})
    
    # compute the mean of the common tokens within each subset
    token_to_resids_1 = dict()
    token_to_resids_2 = dict()

    # Now we'll load each folder and compute the mean of the common tokens
    for folder, tok_id_to_idxs in folder_to_token_id_1.items():
        t1 = load_tensor(os.path.join(folder, f"{comparison['mode1']}.pt"))
        # now we'll collect the means of common tokens 
        for tok_id, idxs in tok_id_to_idxs.items():
            mesh_seq_idx, mesh_layer_idx = np.meshgrid(idxs, layer_idxs, indexing="ij")
            if tok_id not in token_to_resids_1:
                token_to_resids_1[tok_id] = [t1[mesh_seq_idx, mesh_layer_idx, :]]
            else:
                token_to_resids_1[tok_id].append(t1[mesh_seq_idx, mesh_layer_idx, :])
    
    for folder, tok_id_to_idxs in folder_to_token_id_2.items():
        t2 = load_tensor(os.path.join(folder, f"{comparison['mode2']}.pt"))
        for tok_id, idxs in tok_id_to_idxs.items():
            mesh_seq_idx, mesh_layer_idx = np.meshgrid(idxs, layer_idxs, indexing="ij")
            if tok_id not in token_to_resids_2:
                token_to_resids_2[tok_id] = [t2[mesh_seq_idx, mesh_layer_idx, :]]
            else:
                token_to_resids_2[tok_id].append(t2[mesh_seq_idx, mesh_layer_idx, :])

    # collect token resids 
    token_to_resids_1 = {tok_id: np.concatenate(resids, axis=0) for tok_id, resids in token_to_resids_1.items()}
    token_to_resids_2 = {tok_id: np.concatenate(resids, axis=0) for tok_id, resids in token_to_resids_2.items()}
    
    # subtract the means and collect total number of samples in each subset
    num_samples_1 = 0
    num_samples_2 = 0
    for tok_id in token_to_resids_1:
        # compute mu_tot = (mu1 + mu2) / 2
        overall_token_resid_mean = (np.mean(token_to_resids_1[tok_id], axis=0) + np.mean(token_to_resids_2[tok_id], axis=0)) / 2
        token_to_resids_1[tok_id] -= overall_token_resid_mean
        token_to_resids_2[tok_id] -= overall_token_resid_mean

        num_samples_1 += token_to_resids_1[tok_id].shape[0]
        num_samples_2 += token_to_resids_2[tok_id].shape[0]
    
    # next, we need to compute the overall difference of means and the lda direction
    _, num_layers, d_model = token_to_resids_1[next(iter(token_to_resids_1.keys()))].shape
    mean_diff = np.zeros((num_layers, d_model))

    for tok_id in token_to_resids_1:
        mean_diff += np.sum(token_to_resids_1[tok_id], axis=0)/num_samples_1 - np.sum(token_to_resids_2[tok_id], axis=0)/num_samples_2

    # now we'll compute the lda direction
    w_2_to_1, proj1_lda, proj2_lda, (resids_1, resids_2) = lda_per_layer_with_proj(
        data_1=token_to_resids_1,
        data_2=token_to_resids_2,
        reg=0.0
    )

    # additionally, project the data along the mean_diff direction for each layer
    # with shape (num_samples, num_layers, 1)
    proj1_mean_diff = np.einsum('nld,ld->nl', resids_1, mean_diff)[:, :, None]
    proj2_mean_diff = np.einsum('nld,ld->nl', resids_2, mean_diff)[:, :, None]
    
    return {
        'mean_diff': mean_diff,
        'proj1_mean_diff': proj1_mean_diff,
        'proj2_mean_diff': proj2_mean_diff,
        'lda_dir': w_2_to_1,
        'proj1_lda': proj1_lda,
        'proj2_lda': proj2_lda,
    }
    

def compute_raw_diffs(
        data_root: str,
        comparison: dict[str, str],
        layer_idxs: List[int],
        subject: None | str | list[str] = None,
) -> Dict[str, np.ndarray]:
    """
    given a comparison dict, and a data_root, compute the raw diffs

    first extract token_dict_1 and token_dict_2 (not restricting to common tokens)
    each is a dict mapping token_id to a list of (folder, list of tensor_indices)

    return the mean difference and the lda direction for each layer
    """

    # note we restrict to common tokens here since we're doing token normalized comparisons
    token_data_dicts = collect_token_data(
        data_root=data_root,
        mode1=comparison['mode1'], region1=comparison['region1'],
        mode2=comparison['mode2'], region2=comparison['region2'], 
        subject=subject,
        restrict_to_common_tokens=False,
    )

    token_dict_1, token_dict_2 = token_data_dicts['token_dict_1'], token_data_dicts['token_dict_2']

    # note below we don't care about token_id, we just want to collect the 
    # relevant indices for each subset's folders
    # collect a dict mapping folder to a set of indices
    # so that we may easily go folder by folder (loading takes time)
    folder_to_resid_idxs_1 = dict()
    folder_to_resid_idxs_2 = dict()

    for tok_id, folders_indices in token_dict_1.items():
        for folder, indices in folders_indices.items():
            folder_to_resid_idxs_1.setdefault(folder, set()).update(indices)
    for tok_id, folders_indices in token_dict_2.items():  
        for folder, indices in folders_indices.items():
            folder_to_resid_idxs_2.setdefault(folder, set()).update(indices)
    
    # collect the resid tensors for each subset
    resids_1 = []
    resids_2 = []

    # for each folder, load the resid tensor and collect the relevant indices
    for folder, resid_idxs in folder_to_resid_idxs_1.items():
        t1 = load_tensor(os.path.join(folder, f"{comparison['mode1']}.pt"))
        mesh_seq_idx, mesh_layer_idx = np.meshgrid(list(resid_idxs), layer_idxs, indexing="ij")
        resids_1.append(t1[mesh_seq_idx, mesh_layer_idx, :])
    
    for folder, resid_idxs in folder_to_resid_idxs_2.items():
        t2 = load_tensor(os.path.join(folder, f"{comparison['mode2']}.pt"))
        mesh_seq_idx, mesh_layer_idx = np.meshgrid(list(resid_idxs), layer_idxs, indexing="ij")
        resids_2.append(t2[mesh_seq_idx, mesh_layer_idx, :])

    # collect token resids 
    resids_1 = np.concatenate(resids_1, axis=0)
    resids_2 = np.concatenate(resids_2, axis=0)
    
    # compute the mean_diff
    mean_diff = np.mean(resids_1, axis=0) - np.mean(resids_2, axis=0)
    
    # now we'll compute the lda direction
    w_2_to_1, proj1, proj2, _ = lda_per_layer_with_proj(
        data_1=resids_1,
        data_2=resids_2,
        reg=0.0
    )
    
    # additionally, project the data along the mean_diff direction for each layer
    # with shape (num_samples, num_layers, 1)
    proj1_mean_diff = np.einsum('nld,ld->nl', resids_1, mean_diff)[:, :, None]
    proj2_mean_diff = np.einsum('nld,ld->nl', resids_2, mean_diff)[:, :, None]
    
    return {
        'mean_diff': mean_diff,
        'proj1_mean_diff': proj1_mean_diff,
        'proj2_mean_diff': proj2_mean_diff,
        'lda_dir': w_2_to_1,
        'proj1': proj1,
        'proj2': proj2,
    }


def run_analysis(
    data_root: str,
    out_root: str,
    comparisons: List[Dict[str, Any]],
    subjects: Optional[Union[str, List[str]]] = None,
    layer_batch_size: int = 4,
    verbose: bool = True,
    num_layers: int = 32,
):
    """
    Perform analyses one layer at a time with minimal memory usage.

    If comparison['normalize'] is true, perform both token‐normalized and raw diffs.
    If comparison['normalize'] is false, perform only raw diffs.

    layer_batch_size is the number of layers to process at a time.
    There are a total of num_layers layers, and we process them in batches of layer_batch_size.
    The final output combines the results across all layers into:
      • mean_diff_raw: shape (num_layers, d_model)
      • lda_dir_raw:   shape (num_layers, d_model)
      • proj1_raw:     shape (N1, num_layers, 1)
      • proj2_raw:     shape (N2, num_layers, 1)
    If normalized, also:
      • mean_diff_norm:   shape (num_layers, d_model)
      • proj1_mean_diff_norm: shape (N1, num_layers, 1)
      • proj2_mean_diff_norm: shape (N2, num_layers, 1)
      • lda_dir_norm:     shape (num_layers, d_model)
      • proj1_lda_norm:   shape (N1, num_layers, 1)
      • proj2_lda_norm:   shape (N2, num_layers, 1)

    Within out_root, we create one folder per comparison:
      "comparison_{idx}_{subset}_{mode1}_{region1}_vs_{mode2}_{region2}_{subjects}"
    where:
      - idx is the index in the comparisons list (to ensure uniqueness),
      - subset, mode1, region1, mode2, region2 come from the comparison dict,
      - subjects is either a single subject name, or underscore‐joined list, or "all".

    Inside each comparison folder, we save:
      ├─ comparison_config.yaml
      ├─ subjects_used.yaml
      ├─ mean_diff_raw.npy
      ├─ lda_dir_raw.npy
      ├─ proj1_raw.npy
      ├─ proj2_raw.npy
      ├─ [if normalize]
      │   ├─ mean_diff_norm.npy
      │   ├─ proj1_mean_diff_norm.npy
      │   ├─ proj2_mean_diff_norm.npy
      │   ├─ lda_dir_norm.npy
      │   ├─ proj1_lda_norm.npy
      │   └─ proj2_lda_norm.npy

    Args:
        data_root: Directory containing subject_hash folders with model outputs.
        out_root: Directory to save analysis results.
        comparisons: List of comparison‐dicts, each with keys:
            'name', 'mode1', 'region1', 'mode2', 'region2', 'subset', 'normalize'.
        subjects: None | str | list[str]:
            • None → use all subjects.
            • str  → use only that subject.
            • list → use only those subjects.
        layer_batch_size: Number of layers to process per batch (for memory control).
        verbose: If True, show progress bars via tqdm.
        num_layers: Total number of layers in the model (e.g. 32).
    """
    os.makedirs(out_root, exist_ok=True)

    # Prepare subject‐filter argument for lower‐level functions
    if subjects is None:
        subjects_param = None
        subjects_list = None
    elif isinstance(subjects, str):
        subjects_param = subjects
        subjects_list = [subjects]
    else:
        subjects_param = list(subjects)
        subjects_list = subjects_param[:]

    # Iterate over each comparison
    for idx, comp in enumerate(comparisons):
        mode1 = comp["mode1"]
        region1 = comp["region1"]
        mode2 = comp["mode2"]
        region2 = comp["region2"]
        subset = comp["subset"]
        do_normalized = bool(comp.get("normalize", False))

        # Build a subject‐string for naming
        if subjects_list is None:
            subj_str = "all"
        else:
            subj_str = "_".join(subjects_list)

        # Construct folder name:
        # comparison_{idx}_{subset}_{mode1}_{region1}_vs_{mode2}_{region2}_{subjects}
        comp_folder_name = (
            f"comparison_{idx}_"
            f"{subset}_{mode1}_{region1}_vs_{mode2}_{region2}_"
            f"{subj_str}"
        )
        out_dir = os.path.join(out_root, comp_folder_name)
        os.makedirs(out_dir, exist_ok=True)

        # 1) Save the comparison dict (as YAML)
        with open(os.path.join(out_dir, "comparison_config.yaml"), "w", encoding="utf-8") as f:
            yaml.safe_dump(comp, f)

        # 2) Save the subjects used
        with open(os.path.join(out_dir, "subjects_used.yaml"), "w", encoding="utf-8") as f:
            yaml.safe_dump({"subjects": subjects_param}, f)

        # Prepare accumulators
        # Raw diffs
        mean_raw_batches = []
        lda_raw_batches = []
        proj1_raw_batches = []
        proj2_raw_batches = []

        # Normalized diffs (if requested)
        if do_normalized:
            mean_norm_batches = []
            proj1_mean_norm_batches = []
            proj2_mean_norm_batches = []
            lda_norm_batches = []
            proj1_lda_batches = []
            proj2_lda_batches = []

        # Prepare layer indices and batch count
        layer_indices = list(range(num_layers))
        n_batches = (num_layers + layer_batch_size - 1) // layer_batch_size

        iterator = range(n_batches)
        if verbose:
            iterator = tqdm(iterator, desc=f"Comp {idx}: {comp_folder_name}")

        # Process each layer‐batch
        for batch_i in iterator:
            start = batch_i * layer_batch_size
            end = min(start + layer_batch_size, num_layers)
            batch_layer_idxs = layer_indices[start:end]

            # 3) If normalized, run token‐normalized diffs on these layers
            if do_normalized:
                if verbose:
                    print(f"Computing token‐normalized diffs for layers {start} to {end}")

                norm_results = compute_token_normalized_diffs(
                    data_root=data_root,
                    comparison=comp,
                    layer_idxs=batch_layer_idxs,
                    subject=subjects_param,
                )
                # norm_results keys:
                #  'mean_diff', 'proj1_mean_diff', 'proj2_mean_diff',
                #  'lda_dir',  'proj1_lda',       'proj2_lda'
                mean_norm_batches.append(norm_results["mean_diff"])           # (len(batch), d_model)
                proj1_mean_norm_batches.append(norm_results["proj1_mean_diff"])  # (N1, len(batch), 1)
                proj2_mean_norm_batches.append(norm_results["proj2_mean_diff"])  # (N2, len(batch), 1)
                lda_norm_batches.append(norm_results["lda_dir"])             # (len(batch), d_model)
                proj1_lda_batches.append(norm_results["proj1_lda"])          # (N1, len(batch), 1)
                proj2_lda_batches.append(norm_results["proj2_lda"])          # (N2, len(batch), 1)

            # 4) Always run raw diffs on these layers
            if verbose:
                print(f"Computing raw diffs for layers {start} to {end}")
            raw_results = compute_raw_diffs(
                data_root=data_root,
                comparison=comp,
                layer_idxs=batch_layer_idxs,
                subject=subjects_param,
            )
            # raw_results keys: 'mean_diff', 'lda_dir', 'proj1', 'proj2'
            mean_raw_batches.append(raw_results["mean_diff"])   # (len(batch), d_model)
            lda_raw_batches.append(raw_results["lda_dir"])     # (len(batch), d_model)
            proj1_raw_batches.append(raw_results["proj1"])     # (N1, len(batch), 1)
            proj2_raw_batches.append(raw_results["proj2"])     # (N2, len(batch), 1)

        # 5) Concatenate raw diffs along the layer axis
        mean_diff_raw = np.concatenate(mean_raw_batches, axis=0)   # (num_layers, d_model)
        lda_dir_raw   = np.concatenate(lda_raw_batches,   axis=0)   # (num_layers, d_model)
        proj1_raw_all = np.concatenate(proj1_raw_batches, axis=1)   # (N1, num_layers, 1)
        proj2_raw_all = np.concatenate(proj2_raw_batches, axis=1)   # (N2, num_layers, 1)

        # Save raw results
        np.save(os.path.join(out_dir, "mean_diff_raw.npy"), mean_diff_raw)
        np.save(os.path.join(out_dir, "lda_dir_raw.npy"),   lda_dir_raw)
        np.save(os.path.join(out_dir, "proj1_raw.npy"),      proj1_raw_all)
        np.save(os.path.join(out_dir, "proj2_raw.npy"),      proj2_raw_all)

        # 6) If normalized, concatenate and save those too
        if do_normalized:
            mean_diff_norm        = np.concatenate(mean_norm_batches, axis=0)       # (num_layers, d_model)
            proj1_mean_diff_norm  = np.concatenate(proj1_mean_norm_batches, axis=1) # (N1, num_layers, 1)
            proj2_mean_diff_norm  = np.concatenate(proj2_mean_norm_batches, axis=1) # (N2, num_layers, 1)
            lda_dir_norm          = np.concatenate(lda_norm_batches, axis=0)        # (num_layers, d_model)
            proj1_lda_norm        = np.concatenate(proj1_lda_batches, axis=1)       # (N1, num_layers, 1)
            proj2_lda_norm        = np.concatenate(proj2_lda_batches, axis=1)       # (N2, num_layers, 1)

            np.save(os.path.join(out_dir, "mean_diff_norm.npy"),          mean_diff_norm)
            np.save(os.path.join(out_dir, "proj1_mean_diff_norm.npy"),   proj1_mean_diff_norm)
            np.save(os.path.join(out_dir, "proj2_mean_diff_norm.npy"),   proj2_mean_diff_norm)
            np.save(os.path.join(out_dir, "lda_dir_norm.npy"),           lda_dir_norm)
            np.save(os.path.join(out_dir, "proj1_lda_norm.npy"),         proj1_lda_norm)
            np.save(os.path.join(out_dir, "proj2_lda_norm.npy"),         proj2_lda_norm)

        if verbose:
            print(f" → Saved results for comparison '{comp_folder_name}' in {out_dir}")
    
# run the comparisons
if __name__ == "__main__":
    # math specific
    run_analysis(
        data_root="reasoning_resid_data",
        out_root="reasoning_resid_comparisons",
        subjects=['math'],
        comparisons=COMPARISONS,
        layer_batch_size=1,
    )
    # all prompts
    run_analysis(
        data_root="reasoning_resid_data",
        out_root="reasoning_resid_comparisons",
        subjects=None,
        comparisons=COMPARISONS,
        layer_batch_size=1,
    )
