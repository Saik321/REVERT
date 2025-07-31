#!/usr/bin/env python3
"""pySCENIC GRN construction module for the REVERT pipeline.

This script loads a MAGIC-imputed single-cell expression matrix
and applies the pySCENIC workflow to infer regulons and their activity.
Outputs are saved as CSV files that can be read by ``main_REVERT.R``.
"""

from __future__ import annotations

import argparse
import glob
import os
from typing import Iterable

import pandas as pd

# Default paths for running the pipeline without arguments
expr_matrix_path = (
    "C:/Users/saikr/Downloads/REVERT-main/REVERT-main/Result/CRC_organoid_results/magic_expr.csv"
)
adjacencies_path = (
    "C:/Users/saikr/Downloads/REVERT-main/REVERT-main/Result/CRC_organoid_results/pyscenic/adjacencies.csv"
)
motif_table_path = (
    "C:/Users/saikr/Downloads/REVERT-main/REVERT-main/pyscenic_resources/motifs-v9-nr.hgnc-m0.001-o0.0.tbl.txt"
)
rankings_db_path = (
    "C:/Users/saikr/Downloads/REVERT-main/REVERT-main/pyscenic_resources/hg38__refseq-r80__500bp_up_and_100bp_down_tss.mc9nr.genes_vs_motifs.rankings.feather"
)
output_dir = (
    "C:/Users/saikr/Downloads/REVERT-main/REVERT-main/Result/CRC_organoid_results/pyscenic"
)

try:
    from arboreto.algo import grnboost2
    from arboreto.utils import load_tf_names
    from pyscenic.aucell import aucell
    from pyscenic.prune import prune2df, df2regulons
    from pyscenic.rnkdb import FeatherRankingDatabase as RankingDatabase
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError(
        "pySCENIC dependencies are missing. Install pyscenic and arboreto first"
    ) from exc

try:
    from rpy2.robjects import pandas2ri, r
except ImportError:
    pandas2ri = None
    r = None


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def _load_expression(path: str) -> pd.DataFrame:
    """Load an expression matrix saved as CSV or RData."""
    if path.endswith(".RData"):
        if pandas2ri is None or r is None:
            raise ImportError("rpy2 is required to read RData files")
        pandas2ri.activate()
        r["load"](path)
        magic_exp = r["magic.exp"]
        expr = pandas2ri.rpy2py(magic_exp.rx2("result"))
        return pd.DataFrame(expr, index=expr.index.astype(str), columns=expr.columns.astype(str))
    else:
        return pd.read_csv(path, index_col=0)


def _regulons_to_matrix(regulons: Iterable) -> pd.DataFrame:
    """Convert a list of regulons to a binary matrix (gene x regulon)."""
    genes = sorted({g for reg in regulons for g in reg.gene2weight})
    df = pd.DataFrame(0, index=genes, columns=[r.name for r in regulons], dtype=int)
    for reg in regulons:
        df.loc[list(reg.gene2weight.keys()), reg.name] = 1
    return df


# -----------------------------------------------------------------------------
# Main workflow
# -----------------------------------------------------------------------------

def run_pyscenic(
    expr_matrix_path: str,
    adjacencies_path: str,
    motif_tbl: str,
    ranking_db: str,
    output_dir: str,
    n_workers: int = 4,
) -> None:
    """Execute the pySCENIC workflow.

    Parameters
    ----------
    expr_matrix_path : str
        Path to the MAGIC-imputed expression matrix (CSV or ``.RData``).
    adjacencies_path : str
        Output path for the raw GRN adjacencies.
    motif_tbl : str
        Path to the motif annotations table (``motifs.tbl``).
    ranking_db : str
        pySCENIC ranking database (``*.feather``).
    output_dir : str
        Directory where ``regulon_per_cell.csv`` and ``regulon_per_gene.csv``
        will be written.
    n_workers : int, optional
        Number of cores to use, by default 4.
    """
    os.makedirs(output_dir, exist_ok=True)

    expr = _load_expression(expr_matrix_path)
    tf_names = list(expr.columns)

    # Step 1. GRN inference using GRNBoost2
    adjacencies = grnboost2(expr, tf_names=tf_names, n_jobs=n_workers)
    adjacencies.to_csv(adjacencies_path, index=False)

    # Step 2. Prune with motif information
    dbs = [RankingDatabase(fname=ranking_db, name=os.path.splitext(os.path.basename(ranking_db))[0])]
    motifs = prune2df(adjacencies, expr, dbs, motif_tbl, num_workers=n_workers)
    regulons = df2regulons(motifs)

    # Step 3. AUCell to compute regulon activity
    auc_mtx = aucell(expr, regulons, num_workers=n_workers)

    # Step 4. Save outputs
    auc_mtx.to_csv(os.path.join(output_dir, "regulon_per_cell.csv"))
    reg_gene_mat = _regulons_to_matrix(regulons)
    reg_gene_mat.to_csv(os.path.join(output_dir, "regulon_per_gene.csv"))


# -----------------------------------------------------------------------------
# Command-line interface
# -----------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run pySCENIC for the REVERT pipeline")
    parser.add_argument("--expr", default=expr_matrix_path, help="MAGIC-imputed expression matrix")
    parser.add_argument("--adjacencies", default=adjacencies_path, help="Output CSV for adjacencies")
    parser.add_argument("--motif_tbl", default=motif_table_path, help="Motif annotations table")
    parser.add_argument("--rankings", default=rankings_db_path, help="Ranking database (.feather)")
    parser.add_argument("--outdir", default=output_dir, help="Output directory for pySCENIC results")
    parser.add_argument("--n_workers", type=int, default=4, help="Number of worker processes")
    return parser.parse_args()


if __name__ == "__main__":  # pragma: no cover
    args = _parse_args()
    run_pyscenic(
        args.expr,
        args.adjacencies,
        args.motif_tbl,
        args.rankings,
        args.outdir,
        args.n_workers,
    )
