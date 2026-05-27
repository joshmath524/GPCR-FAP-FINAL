"""
Command-line interface for GPCR Class A Functional Activity prediction.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

from .predict import predict_single, predict_batch, load_predictor


def main():
    parser = argparse.ArgumentParser(
        description="GPCR Class A Functional Activity Prediction CLI"
    )
    parser.add_argument(
        "--receptor",
        type=str,
        help="GPCR Class A receptor name (e.g., ADRB2)",
    )
    parser.add_argument(
        "--ligand",
        type=str,
        help="Ligand SMILES string",
    )
    parser.add_argument(
        "--smiles",
        nargs="+",
        help="One or more SMILES strings (requires --receptor)",
    )
    parser.add_argument(
        "--input",
        type=str,
        help="Input CSV file with 'receptor' and 'ligand' (or 'smiles') columns",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="gpcr_predictions.csv",
        help="Output CSV file path",
    )
    parser.add_argument(
        "--artifact-dir",
        type=str,
        default=".",
        help="Directory containing artifacts/ folder",
    )
    
    args = parser.parse_args()
    
    # Load predictor
    try:
        predictor = load_predictor(args.artifact_dir)
    except Exception as e:
        print(f"Error loading predictor: {e}", file=sys.stderr)
        sys.exit(1)
    
    results = []
    
    if args.input:
        # Batch mode from CSV
        try:
            df = pd.read_csv(args.input)
        except Exception as e:
            print(f"Error reading CSV: {e}", file=sys.stderr)
            sys.exit(1)
        
        # Find receptor and ligand columns
        receptor_col = next(
            (c for c in df.columns if c.lower() in ("receptor", "receptor_name", "gpcr")),
            None
        )
        ligand_col = next(
            (c for c in df.columns if c.lower() in ("ligand", "smiles", "canonical_smiles", "smi")),
            None
        )
        
        if receptor_col is None:
            print("Error: CSV must have a 'receptor' column", file=sys.stderr)
            sys.exit(1)
        if ligand_col is None:
            print("Error: CSV must have a 'ligand' or 'smiles' column", file=sys.stderr)
            sys.exit(1)
        
        pairs = list(zip(df[receptor_col].astype(str), df[ligand_col].astype(str)))
        results = predict_batch(pairs, predictor=predictor)
        
        # Create output DataFrame
        df_out = df.copy()
        df_out["predicted_class"] = [r.predicted_class for r in results]
        df_out["class_id"] = [r.class_id for r in results]
        df_out["prob_agonist"] = [r.prob_agonist for r in results]
        df_out["prob_antagonist"] = [r.prob_antagonist for r in results]
        df_out["prob_inactive"] = [r.prob_inactive for r in results]
        df_out["prob_std_error"] = [
            f"{r.prob_std_error:.6f}" if r.prob_std_error is not None else ""
            for r in results
        ]
        df_out["prob_std_error_pct"] = [
            f"{r.prob_std_error * 100:.2f}%" if r.prob_std_error is not None else ""
            for r in results
        ]
        df_out["canonical_smiles"] = [r.canonical_smiles for r in results]
        df_out["error"] = [r.error for r in results]
        
    elif args.receptor and args.ligand:
        # Single prediction
        result = predict_single(args.receptor, args.ligand, predictor=predictor)
        results = [result]
        
        # Create output DataFrame
        df_out = pd.DataFrame([{
            "receptor": result.receptor,
            "ligand_smiles": result.ligand_smiles,
            "canonical_smiles": result.canonical_smiles,
            "predicted_class": result.predicted_class,
            "class_id": result.class_id,
            "prob_agonist": result.prob_agonist,
            "prob_antagonist": result.prob_antagonist,
            "prob_inactive": result.prob_inactive,
            "prob_std_error": f"{result.prob_std_error:.6f}" if result.prob_std_error else "",
            "prob_std_error_pct": f"{result.prob_std_error * 100:.2f}%" if result.prob_std_error else "",
            "error": result.error,
        }])
        
    elif args.receptor and args.smiles:
        # Multiple ligands for one receptor
        pairs = [(args.receptor, smiles) for smiles in args.smiles]
        results = predict_batch(pairs, predictor=predictor)
        
        df_out = pd.DataFrame([{
            "receptor": r.receptor,
            "ligand_smiles": r.ligand_smiles,
            "canonical_smiles": r.canonical_smiles,
            "predicted_class": r.predicted_class,
            "class_id": r.class_id,
            "prob_agonist": r.prob_agonist,
            "prob_antagonist": r.prob_antagonist,
            "prob_inactive": r.prob_inactive,
            "prob_std_error": f"{r.prob_std_error:.6f}" if r.prob_std_error else "",
            "prob_std_error_pct": f"{r.prob_std_error * 100:.2f}%" if r.prob_std_error else "",
            "error": r.error,
        } for r in results])
        
    else:
        parser.print_help()
        sys.exit(1)
    
    # Write output
    try:
        df_out.to_csv(args.output, index=False)
        print(f"Predictions written to {args.output}")
    except Exception as e:
        print(f"Error writing output: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
