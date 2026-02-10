#!/usr/bin/env python3
"""Run batch prediction and evaluation for multiple configurations."""

import subprocess
import sys
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Run batch prediction and evaluation.")
    parser.add_argument("--vars", nargs="+", default=["pr"], help="Variables to process (e.g., pr tas)")
    parser.add_argument("--domains", nargs="+", default=["ALPS"], help="Domains to process (e.g., ALPS NZ SA)")
    parser.add_argument("--exps", nargs="+", default=["ESD_pseudo_reality"], help="Training experiments to process")
    parser.add_argument("--use-orography", action="store_true", help="Use models trained with orography as co-variable")
    
    args = parser.parse_args()

    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    predict_script = os.path.join(script_dir, "predict_validation.py")
    eval_script = os.path.join(script_dir, "..", "eval", "run_all_eval.py")

    # Ensure paths are absolute
    predict_script = os.path.abspath(predict_script)
    eval_script = os.path.abspath(eval_script)

    for exp in args.exps:
        for domain in args.domains:
            for var in args.vars:
                print(f"\n{'='*60}")
                print(f"Processing: Exp={exp}, Domain={domain}, Var={var}")
                print(f"{'='*60}")

                # 1. Run prediction
                print(f"\n[1/2] Running prediction: {os.path.basename(predict_script)}")
                pred_args = [sys.executable, predict_script, var, domain, exp]
                if args.use_orography:
                    pred_args.append("true")
                try:
                    subprocess.run(pred_args, check=True)
                except subprocess.CalledProcessError as e:
                    print(f"Error running prediction for {var} {domain} {exp}: {e}")
                    continue

                # 2. Run evaluation
                print(f"\n[2/2] Running evaluation: {os.path.basename(eval_script)}")
                env = os.environ.copy()
                env["VAR_TARGET"] = var
                env["DOMAIN"] = domain
                env["TRAINING_EXPERIMENT"] = exp
                env["USE_OROGRAPHY"] = "true" if args.use_orography else "false"
                
                try:
                    subprocess.run([sys.executable, eval_script], env=env, check=True)
                except subprocess.CalledProcessError as e:
                    print(f"Error running evaluation for {var} {domain} {exp}: {e}")
                    continue

    print("\nBatch processing completed!")

if __name__ == "__main__":
    main()
