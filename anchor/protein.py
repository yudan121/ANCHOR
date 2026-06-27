from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd


def protein_obsm_to_frame(adata: ad.AnnData, key: str) -> pd.DataFrame:
    protein = adata.obsm[key]
    if isinstance(protein, pd.DataFrame):
        df = protein.copy()
        df.index = adata.obs_names
        df.columns = df.columns.astype(str)
        return df
    names = [str(x) for x in adata.uns.get("protein_names", adata.uns.get(f"{key}_names", range(protein.shape[1])))]
    return pd.DataFrame(np.asarray(protein, dtype=np.float32), index=adata.obs_names, columns=names)


def build_protein_feature_tables(
    protein_raw: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    from .teacher import fit_protein_teacher_stats

    protein_names = protein_raw.columns.astype(str).tolist()
    teacher_stats = fit_protein_teacher_stats(protein_raw, protein_names)
    kappa = np.asarray(teacher_stats["kappa"], dtype=np.float32)
    median = np.asarray(teacher_stats["median"], dtype=np.float32)
    mad = np.asarray(teacher_stats["mad"], dtype=np.float32)
    raw_np = protein_raw.to_numpy(dtype=np.float32)
    arcsinh_np = np.arcsinh(raw_np / kappa.reshape(1, -1)).astype(np.float32)
    teacher_z_np = ((arcsinh_np - median.reshape(1, -1)) / mad.reshape(1, -1)).astype(np.float32)
    return {
        "raw": protein_raw.astype(np.float32).copy(),
        "arcsinh": pd.DataFrame(arcsinh_np, index=protein_raw.index, columns=protein_names),
        "teacher_z": pd.DataFrame(teacher_z_np, index=protein_raw.index, columns=protein_names),
    }
