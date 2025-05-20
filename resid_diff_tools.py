import os
import torch
import yaml
import numpy as np
from typing import List, Dict, Any, Set, Optional
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

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
    except Exception:
        return None


def load_metadata(path: str) -> Optional[List[Dict[str, Any]]]:
    """Load YAML metadata, return None on failure."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception:
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
    """Collect token_ids that appear in region1 of mode1 AND region2 of mode2 across all prompts."""
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
    return set1 & set2

# -----------------------------------------
# Single-layer analysis helpers
# -----------------------------------------
def token_normalize_layer(resid_layer: np.ndarray,
                           metadata: List[Dict[str, Any]],
                           idx1: List[int],
                           idx2: List[int],
                           valid_ids: Set[int]) -> np.ndarray:
    """Token-normalize at single-layer: subtract mu_token=(mu1+mu2)/2 for tokens in both regions."""
    resid_norm = resid_layer.copy()
    positions: Dict[int, Dict[str, List[int]]] = {}
    for idx in idx1:
        tok_id = metadata[idx].get('token_id', -1)
        if tok_id in valid_ids:
            positions.setdefault(tok_id, {'idx1': [], 'idx2': []})['idx1'].append(idx)
    for idx in idx2:
        tok_id = metadata[idx].get('token_id', -1)
        if tok_id in valid_ids:
            positions.setdefault(tok_id, {'idx1': [], 'idx2': []})['idx2'].append(idx)
    for tok_id, parts in positions.items():
        idxs1, idxs2 = parts['idx1'], parts['idx2']
        if not idxs1 or not idxs2:
            continue
        mu1 = np.nanmean(resid_norm[idxs1], axis=0)
        mu2 = np.nanmean(resid_norm[idxs2], axis=0)
        mu_token = 0.5 * (mu1 + mu2)
        for pos in idxs1 + idxs2:
            resid_norm[pos] -= mu_token
    return resid_norm

# -----------------------------------------
# Main analysis: per-layer, per-comparison, per-scope
# -----------------------------------------

def mean_diff_layer(resid1_layer: np.ndarray, idx1: List[int],
                     resid2_layer: np.ndarray, idx2: List[int]) -> np.ndarray:
    """Mean difference at a single layer."""
    return resid1_layer[idx1].mean(axis=0) - resid2_layer[idx2].mean(axis=0)


def lda_dir_layer(resid1_layer: np.ndarray, idx1: List[int],
                  resid2_layer: np.ndarray, idx2: List[int]) -> np.ndarray:
    """LDA direction at a single layer."""
    X = np.concatenate([resid1_layer[idx1], resid2_layer[idx2]], axis=0)
    y = np.hstack([np.zeros(len(idx1)), np.ones(len(idx2))])
    lda = LinearDiscriminantAnalysis(solver='eigen')
    lda.fit(X, y)
    return lda.coef_[0]

# -----------------------------------------
# Main analysis: per-layer, per-comparison, per-scope
# -----------------------------------------

def run_analysis(
    data_root: str,
    out_root: str,
    per_subject: bool = False
):
    """
    Perform analyses one layer at a time to reduce memory usage.

    - Scopes: 'ALL' and optionally per-subject.
    - For each comparison and each layer:
        * aggregate idx1_all, idx2_all
        * load per-layer residuals folder-by-folder
        * optionally mask & normalize
        * compute mean diff & LDA dir
        * save arrays of shape (n_layers, d_model)
    """
    os.makedirs(out_root, exist_ok=True)
    # Discover prompt folders and subjects
    prompt_folders = [os.path.join(data_root, d) for d in sorted(os.listdir(data_root))
                      if os.path.isdir(os.path.join(data_root, d))]
    subject_map: Dict[str, List[str]] = {}
    for fld in prompt_folders:
        subj = os.path.basename(fld).split('_',1)[0]
        subject_map.setdefault(subj, []).append(fld)
    scopes = {'ALL': prompt_folders}
    if per_subject:
        for subj, fl in subject_map.items(): scopes[subj] = fl

    for scope, folders in scopes.items():
        for comp in COMPARISONS:
            name = comp['name']; m1, r1 = comp['mode1'], comp['region1']
            m2, r2 = comp['mode2'], comp['region2']
            subset, normalize = comp['subset'], comp['normalize']
            # Precompute metadata and idx lists
            meta_all = []
            idx1_all, idx2_all = [], []
            offsets = []
            total1 = 0
            for fld in folders:
                m1_meta = load_metadata(os.path.join(fld, f"{m1}_metadata.yaml")) or []
                m2_meta = load_metadata(os.path.join(fld, f"{m2}_metadata.yaml")) or []
                idx1 = filter_indices(m1_meta, r1, subset)
                idx2 = filter_indices(m2_meta, r2, subset)
                if not idx1 or not idx2: continue
                # record offsets for this folder
                offsets.append((fld, total1))
                idx1_all += [total1 + i for i in idx1]
                total1 += len(m1_meta)
                idx2_all += [total1 + i for i in idx2]
                total1 += len(m2_meta)
                meta_all.extend(m1_meta)
                meta_all.extend(m2_meta)
            if not idx1_all: continue
            # Determine layers & d_model from first tensor
            sample = load_tensor(os.path.join(folders[0], f"{m1}.pt"))
            if sample is None: continue
            n_layers, d_model = sample.shape[1], sample.shape[2]
            mean_diff = np.zeros((n_layers, d_model))
            lda_dir   = np.zeros((n_layers, d_model))

            # Precompute valid_ids if needed
            if normalize:
                valid_ids = compute_common_tokens(data_root, m1, r1, m2, r2)

            # Loop per layer
            for l in range(n_layers):
                # aggregate per-layer residuals
                resid1_layer_list, resid2_layer_list = [], []
                for fld in folders:
                    arr1 = load_tensor(os.path.join(fld, f"{m1}.pt"))
                    arr2 = load_tensor(os.path.join(fld, f"{m2}.pt"))
                    if arr1 is None or arr2 is None: continue
                    resid1_layer_list.append(arr1[:, l, :])
                    resid2_layer_list.append(arr2[:, l, :])
                r1_layer = np.concatenate(resid1_layer_list, axis=0)
                r2_layer = np.concatenate(resid2_layer_list, axis=0)
                # mask & normalize if required
                if normalize:
                    # mask invalid
                    for i, m in enumerate(meta1_all):
                        if m.get('token_id', -1) not in valid_ids and i < r1_layer.shape[0]:
                            r1_layer[i] = np.nan
                    for j, m in enumerate(meta2_all):
                        if m.get('token_id', -1) not in valid_ids and j < r2_layer.shape[0]:
                            r2_layer[j] = np.nan
                    # combine & normalize
                    big = np.concatenate([r1_layer, r2_layer], axis=0)
                    combined_idx1 = idx1_all
                    combined_idx2 = [i + r1_layer.shape[0] for i in idx2_all]
                    normed = token_normalize_layer(big, meta_all, combined_idx1, combined_idx2, valid_ids)
                    r1_layer = normed[:r1_layer.shape[0]]
                    r2_layer = normed[r1_layer.shape[0]:]
                # compute valid per-layer indices
                idx1_layer = [i for i in idx1_all if i < r1_layer.shape[0]]
                idx2_layer = [i for i in idx2_all if i < r2_layer.shape[0]]
                # compute for this layer
                mean_diff[l] = mean_diff_layer(r1_layer, idx1_layer, r2_layer, idx2_layer)
                lda_dir[l]   = lda_dir_layer  (r1_layer, idx1_layer, r2_layer, idx2_layer)

            # save
            outd = os.path.join(out_root, scope, name)
            os.makedirs(outd, exist_ok=True)
            np.save(os.path.join(outd, 'mean_diff.npy'), mean_diff)
            np.save(os.path.join(outd, 'lda_dir.npy'),   lda_dir)

    print("Analysis complete. Results saved in", out_root)



# run the comparisons
if __name__ == "__main__":
    run_analysis(
        data_root="reasoning_resid_data",
        out_root="resid_comparisons",
        per_subject=['math']
    )