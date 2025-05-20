import os
import torch
import yaml
import numpy as np
from typing import List, Dict, Any, Set, Optional, Tuple
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import time
from datetime import timedelta

# -----------------------------------------
# Configuration of comparisons
# -----------------------------------------
COMPARISONS = [
    {"name": "base_answer_vs_enhanced_reason_body",       "mode1": "base",              "region1": "answering",    "mode2": "reasoning_boosted","region2": "reasoning", "subset": "body",    "normalize": True},
    {"name": "base_reason_vs_base_answer_body",           "mode1": "base",              "region1": "reasoning",    "mode2": "base",              "region2": "answering",  "subset": "body",    "normalize": True},
    {"name": "enhanced_reason_vs_enhanced_answer_body",   "mode1": "reasoning_boosted","region1": "reasoning",    "mode2": "reasoning_boosted","region2": "answering", "subset": "body",    "normalize": True},
    {"name": "base_reason_vs_immediate_answer_body",      "mode1": "base",              "region1": "reasoning",    "mode2": "immediate_answer","region2": "answering", "subset": "body",    "normalize": True},
    {"name": "enhanced_reason_vs_base_reason_body",       "mode1": "reasoning_boosted","region1": "reasoning",    "mode2": "base",              "region2": "reasoning",  "subset": "body",    "normalize": True},
    {"name": "base_reason_vs_base_answer_initial",        "mode1": "base",              "region1": "reasoning",    "mode2": "base",              "region2": "answering",  "subset": "initial", "normalize": False},
    {"name": "enhanced_reason_vs_enhanced_answer_initial","mode1": "reasoning_boosted","region1": "reasoning",    "mode2": "reasoning_boosted","region2": "answering", "subset": "initial", "normalize": False},
    {"name": "enhanced_reason_vs_base_reason_initial",    "mode1": "reasoning_boosted","region1": "reasoning",    "mode2": "base",              "region2": "reasoning",  "subset": "initial", "normalize": False},
    {"name": "base_reason_vs_immediate_answer_initial",   "mode1": "base",              "region1": "reasoning",    "mode2": "immediate_answer","region2": "answering", "subset": "initial", "normalize": False},
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


def filter_indices(metadata: List[Dict[str, Any]], region: str, subset: str) -> List[int]:
    """Filter by region, subset, and token_id <128000."""
    idxs = []
    for i, m in enumerate(metadata):
        if m.get('region') != region: continue
        tok_id = m.get('token_id', -1)
        if tok_id < 0 or tok_id >= 128000: continue
        dist = m.get('dist_from_prev_marker')
        if subset == 'body' and dist is not None and dist > 1:
            idxs.append(i)
        elif subset == 'initial' and dist == 1:
            idxs.append(i)
    return idxs


def compute_common_tokens(data_root: str,
                          mode1: str, region1: str,
                          mode2: str, region2: str) -> Set[int]:
    """Collect token_ids that appear in both mode1/region1 and mode2/region2 across all prompts."""
    set1, set2 = set(), set()
    for d in sorted(os.listdir(data_root)):
        folder = os.path.join(data_root, d)
        if not os.path.isdir(folder): continue
        meta1 = load_metadata(os.path.join(folder, f"{mode1}_metadata.yaml")) or []
        meta2 = load_metadata(os.path.join(folder, f"{mode2}_metadata.yaml")) or []
        for m in meta1:
            if m.get('region')==region1 and 0<=m.get('token_id',-1)<128000:
                set1.add(m['token_id'])
        for m in meta2:
            if m.get('region')==region2 and 0<=m.get('token_id',-1)<128000:
                set2.add(m['token_id'])
    common = set1 & set2
    print(f"Common tokens between {mode1}/{region1} and {mode2}/{region2}: {len(common)}")
    return common

# -----------------------------------------
# Single-layer helpers
# -----------------------------------------

def token_normalize_layer(resid_layer: np.ndarray,
                           metadata: List[Dict[str, Any]],
                           idx1: List[int],
                           idx2: List[int],
                           valid_ids: Set[int]) -> np.ndarray:
    """Token-normalize at single-layer: subtract mu_token=(mu1+mu2)/2 for tokens in both regions."""
    resid_norm = resid_layer.copy()
    positions: Dict[int, Dict[str, List[int]]] = {}
    
    # Group indices by token_id
    for idx in idx1:
        if idx >= len(metadata): continue  # Skip if out of bounds
        tok_id = metadata[idx].get('token_id', -1)
        if tok_id in valid_ids:
            positions.setdefault(tok_id, {'idx1': [], 'idx2': []})['idx1'].append(idx)
    
    for idx in idx2:
        if idx >= len(metadata): continue  # Skip if out of bounds
        tok_id = metadata[idx].get('token_id', -1)
        if tok_id in valid_ids:
            positions.setdefault(tok_id, {'idx1': [], 'idx2': []})['idx2'].append(idx)
    
    # Apply normalization for each token type
    for tok_id, parts in positions.items():
        idxs1, idxs2 = parts['idx1'], parts['idx2']
        if not idxs1 or not idxs2: continue
        
        # Use nanmean to handle any NaN values safely
        mu1 = np.nanmean(resid_norm[idxs1], axis=0)
        mu2 = np.nanmean(resid_norm[idxs2], axis=0)
        mu_token = 0.5 * (mu1 + mu2)
        
        # Apply normalization
        for pos in idxs1 + idxs2:
            resid_norm[pos] -= mu_token
    
    return resid_norm


def mean_diff_layer(resid1: np.ndarray, idx1: List[int],
                     resid2: np.ndarray, idx2: List[int]) -> np.ndarray:
    """Mean difference at a single layer."""
    # Use nanmean to safely handle NaN values
    return np.nanmean(resid1[idx1], axis=0) - np.nanmean(resid2[idx2], axis=0)


def lda_dir_layer(resid1: np.ndarray, idx1: List[int],
                  resid2: np.ndarray, idx2: List[int]) -> np.ndarray:
    """LDA direction at a single layer."""
    # Filter out indices that might point to NaN values
    valid_idx1 = [i for i in idx1 if not np.any(np.isnan(resid1[i]))]
    valid_idx2 = [i for i in idx2 if not np.any(np.isnan(resid2[i]))]
    
    # Ensure we have enough samples
    if len(valid_idx1) < 2 or len(valid_idx2) < 2:
        print(f"Warning: Not enough valid samples for LDA. Using zeros. Valid counts: {len(valid_idx1)}, {len(valid_idx2)}")
        return np.zeros(resid1.shape[1])
    
    # Compute LDA direction
    X = np.concatenate([resid1[valid_idx1], resid2[valid_idx2]], axis=0)
    y = np.hstack([np.zeros(len(valid_idx1)), np.ones(len(valid_idx2))])
    
    try:
        lda = LinearDiscriminantAnalysis(solver='eigen')
        lda.fit(X, y)
        return lda.coef_[0]
    except Exception as e:
        print(f"LDA computation error: {e}. Using zeros.")
        return np.zeros(resid1.shape[1])

# -----------------------------------------
# Simplified approach - process tensors directly
# -----------------------------------------

def run_analysis(
    data_root: str,
    out_root: str,
    per_subject: bool = False,
    verbose: bool = True
):
    """
    Perform analyses one layer at a time with minimal memory usage.
    
    Args:
        data_root: Directory containing subject_hash folders with model outputs
        out_root: Directory to save analysis results
        per_subject: If True, run separate analyses for each subject
        verbose: If True, print detailed progress information
    """
    start_time = time.time()
    
    # Setup output directory
    os.makedirs(out_root, exist_ok=True)
    print(f"Starting analysis. Data from: {data_root}")
    print(f"Results will be saved to: {out_root}")
    
    # Discover prompt folders and subjects
    prompt_folders = [os.path.join(data_root, d) for d in sorted(os.listdir(data_root))
                      if os.path.isdir(os.path.join(data_root, d))]
    
    if not prompt_folders:
        print(f"Error: No folders found in {data_root}")
        return
        
    print(f"Found {len(prompt_folders)} prompt folders")
    
    # Map subjects
    subject_map: Dict[str, List[str]] = {}
    for fld in prompt_folders:
        subj = os.path.basename(fld).split('_', 1)[0]
        subject_map.setdefault(subj, []).append(fld)
    
    # Define analysis scopes
    scopes = {'ALL': prompt_folders}
    if per_subject:
        for subj, fl in subject_map.items():
            scopes[subj] = fl
            
    print(f"Analysis scopes: {', '.join(scopes.keys())}")

    # Process each scope and comparison
    for scope_idx, (scope, folders) in enumerate(scopes.items()):
        print(f"\nProcessing scope {scope_idx+1}/{len(scopes)}: {scope} ({len(folders)} folders)")
        
        for comp_idx, comp in enumerate(COMPARISONS):
            comp_start = time.time()
            name = comp['name']
            m1, r1 = comp['mode1'], comp['region1']
            m2, r2 = comp['mode2'], comp['region2']
            subset, normalize = comp['subset'], comp['normalize']
            
            print(f"  Comparison {comp_idx+1}/{len(COMPARISONS)}: {name}")
            print(f"    {m1}/{r1} vs {m2}/{r2}, subset={subset}, normalize={normalize}")
            
            # Collect valid folders with their filtered indices
            valid_folders_info = []
            
            for fld in folders:
                m1_path = os.path.join(fld, f"{m1}_metadata.yaml")
                m2_path = os.path.join(fld, f"{m2}_metadata.yaml")
                
                m1_meta = load_metadata(m1_path)
                m2_meta = load_metadata(m2_path)
                
                if m1_meta is None or m2_meta is None:
                    continue
                    
                idx1 = filter_indices(m1_meta, r1, subset)
                idx2 = filter_indices(m2_meta, r2, subset)
                
                if not idx1 or not idx2:
                    if verbose:
                        print(f"    Skipping {os.path.basename(fld)}: No matching indices")
                    continue
                
                # Store valid folder information
                valid_folders_info.append((fld, m1_meta, m2_meta, idx1, idx2))
            
            if not valid_folders_info:
                print(f"    Skipping comparison - no valid folders found")
                continue
                
            print(f"    Processing {len(valid_folders_info)} valid folders")
            
            # Get model dimensions from first valid folder
            sample1_path = os.path.join(valid_folders_info[0][0], f"{m1}.pt")
            sample1 = load_tensor(sample1_path)
            if sample1 is None:
                print(f"    Error: Could not load sample tensor from {sample1_path}")
                continue
                
            n_layers, d_model = sample1.shape[1], sample1.shape[2]
            print(f"    Model dimensions: {n_layers} layers, {d_model} hidden size")
            
            # Initialize output arrays
            mean_diff = np.zeros((n_layers, d_model))
            lda_dir = np.zeros((n_layers, d_model))
            
            # Precompute valid token IDs if normalizing
            valid_ids = None
            if normalize:
                valid_ids = compute_common_tokens(data_root, m1, r1, m2, r2)
                if not valid_ids:
                    print(f"    Warning: No common tokens found for normalization")
            
            # Process each layer
            for l in range(n_layers):
                if verbose or (l % max(1, n_layers // 10) == 0):
                    layer_progress = (l+1)/n_layers * 100
                    elapsed = time.time() - comp_start
                    eta = (elapsed / (l+1)) * (n_layers - l - 1) if l > 0 else 0
                    print(f"    Layer {l+1}/{n_layers} ({layer_progress:.1f}%) - "
                          f"Elapsed: {timedelta(seconds=int(elapsed))}, "
                          f"ETA: {timedelta(seconds=int(eta))}")
                
                r1_all = []  # Will store (tensor, indices) for first condition
                r2_all = []  # Will store (tensor, indices) for second condition
                
                # Process each folder independently
                for fld, m1_meta, m2_meta, idx1, idx2 in valid_folders_info:
                    t1 = load_tensor(os.path.join(fld, f"{m1}.pt"))
                    t2 = load_tensor(os.path.join(fld, f"{m2}.pt"))
                    
                    if t1 is None or t2 is None:
                        continue
                        
                    # Extract layer and apply token normalization if needed
                    r1 = t1[:, l, :]
                    r2 = t2[:, l, :]
                    
                    if normalize and valid_ids:
                        # Mark invalid tokens with NaN
                        r1_norm = r1.copy()
                        r2_norm = r2.copy()
                        
                        for i, meta in enumerate(m1_meta):
                            if meta.get('token_id', -1) not in valid_ids:
                                r1_norm[i] = np.nan
                                
                        for i, meta in enumerate(m2_meta):
                            if meta.get('token_id', -1) not in valid_ids:
                                r2_norm[i] = np.nan
                        
                        # Combine and normalize (per folder)
                        combined = np.concatenate([r1_norm, r2_norm], axis=0)
                        combined_meta = m1_meta + m2_meta
                        combined_idx1 = idx1
                        combined_idx2 = [i + len(m1_meta) for i in idx2]
                        
                        normed = token_normalize_layer(combined, combined_meta, 
                                                      combined_idx1, combined_idx2, valid_ids)
                        
                        r1 = normed[:len(m1_meta)]
                        r2 = normed[len(m1_meta):]
                    
                    # Store normalized tensors with their indices
                    r1_all.append((r1, idx1))
                    r2_all.append((r2, idx2))
                
                # Compute mean difference and LDA direction
                if r1_all and r2_all:
                    # Concatenate all tensors and adjust indices
                    all_r1 = []
                    all_r2 = []
                    all_idx1 = []
                    all_idx2 = []
                    
                    offset1 = 0
                    for r1, idx1 in r1_all:
                        all_r1.append(r1)
                        all_idx1.extend([i + offset1 for i in idx1])
                        offset1 += r1.shape[0]
                    
                    offset2 = 0
                    for r2, idx2 in r2_all:
                        all_r2.append(r2)
                        all_idx2.extend([i + offset2 for i in idx2])
                        offset2 += r2.shape[0]
                    
                    r1_layer = np.concatenate(all_r1, axis=0)
                    r2_layer = np.concatenate(all_r2, axis=0)
                    
                    # Compute metrics
                    mean_diff[l] = mean_diff_layer(r1_layer, all_idx1, r2_layer, all_idx2)
                    lda_dir[l] = lda_dir_layer(r1_layer, all_idx1, r2_layer, all_idx2)
            
            # Save results
            out_dir = os.path.join(out_root, scope, name)
            os.makedirs(out_dir, exist_ok=True)
            
            np.save(os.path.join(out_dir, 'mean_diff.npy'), mean_diff)
            np.save(os.path.join(out_dir, 'lda_dir.npy'), lda_dir)
            
            # Save metadata about the analysis
            with open(os.path.join(out_dir, 'analysis_info.yaml'), 'w') as f:
                info = {
                    'mode1': m1, 'region1': r1,
                    'mode2': m2, 'region2': r2,
                    'subset': subset,
                    'normalize': normalize,
                    'folders_processed': len(valid_folders_info),
                    'shape': [n_layers, d_model]
                }
                yaml.dump(info, f)
            
            comp_time = time.time() - comp_start
            print(f"    Completed in {timedelta(seconds=int(comp_time))}")
    
    # Final stats
    total_time = time.time() - start_time
    print(f"\nAnalysis complete. Total time: {timedelta(seconds=int(total_time))}")
    print(f"Results saved in {out_root}")
    
# run the comparisons
if __name__ == "__main__":
    run_analysis(
        data_root="reasoning_resid_data",
        out_root="resid_comparisons",
        per_subject=['math']
    )