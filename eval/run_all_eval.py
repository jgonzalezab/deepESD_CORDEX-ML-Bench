"""Run all evaluation scripts and merge PDFs into a single report."""

import os
import subprocess
import sys

# Parameters - change these as needed or use environment variables
params = {
    'VAR_TARGET': os.environ.get('VAR_TARGET', 'pr'),
    'TRAINING_EXPERIMENT': os.environ.get('TRAINING_EXPERIMENT', 'ESD_pseudo_reality'),
    'DOMAIN': os.environ.get('DOMAIN', 'ALPS'),
    'USE_OROGRAPHY': os.environ.get('USE_OROGRAPHY', 'false').lower() in ('true', '1', 'yes', 'on')
}

# Paths to the scripts
eval_dir = os.path.dirname(os.path.abspath(__file__))
scripts = [
    os.path.join(eval_dir, 'standard_metrics.py'),
    os.path.join(eval_dir, 'psd_comparison.py'),
    os.path.join(eval_dir, 'daily_comparison.py'),
    os.path.join(eval_dir, 'dist_comparison.py')
]


def merge_pdfs(pdf_list, output_path):
    """Merge multiple PDFs into a single file."""
    try:
        from pypdf import PdfWriter
    except ImportError:
        try:
            from PyPDF2 import PdfWriter
        except ImportError:
            print("\n[ERROR] Could not find 'pypdf' or 'PyPDF2' library.")
            print("Please install one of them to merge the PDFs:")
            print("  pip install pypdf")
            return False

    writer = PdfWriter()
    for pdf in pdf_list:
        if os.path.exists(pdf):
            writer.append(pdf)
        else:
            print(f"Warning: PDF file not found and skipped: {pdf}")
    
    with open(output_path, "wb") as f:
        writer.write(f)
    print(f"\nSuccessfully merged PDFs into: {output_path}")
    return True


def main():
    """Run all evaluation scripts and merge the results."""
    # Update environment variables
    env = os.environ.copy()
    env.update({k: str(v).lower() if k == 'USE_OROGRAPHY' else str(v) for k, v in params.items()})

    output_pdfs = []
    
    # Model name for file naming
    model_name = f"DeepESD_{params['TRAINING_EXPERIMENT']}_{params['DOMAIN']}_{params['VAR_TARGET']}"
    
    # Get figs path
    sys.path.append(os.path.join(eval_dir, '..', 'src'))
    from config import FIGS_PATH
    
    # 1. Run scripts
    for script in scripts:
        print(f"\n--- Running {os.path.basename(script)} ---")
        try:
            subprocess.run([sys.executable, script], env=env, check=True)
            
            # Construct the expected PDF path (dist_comparison produces no PDF)
            if 'standard_metrics' in script:
                pdf_name = f"{model_name}_metrics_report.pdf"
            elif 'psd_comparison' in script:
                pdf_name = f"{model_name}_psd_comparison.pdf"
            elif 'daily_comparison' in script:
                pdf_name = f"{model_name}_daily_comparison.pdf"
            else:
                pdf_name = None  # e.g. dist_comparison
            if pdf_name is not None:
                output_pdfs.append(os.path.join(FIGS_PATH, pdf_name))
            
        except subprocess.CalledProcessError as e:
            print(f"Error running {script}: {e}")
            sys.exit(1)

    # 2. Merge resulting PDFs
    merged_output = os.path.join(FIGS_PATH, f"{model_name}_FULL_REPORT.pdf")
    if merge_pdfs(output_pdfs, merged_output):
        # 3. Remove the temporary PDFs
        for pdf in output_pdfs:
            if os.path.exists(pdf):
                os.remove(pdf)
        print(f"\nFull evaluation report: {merged_output}")
    else:
        print("\nPDF merging failed. Individual reports are available:")
        for pdf in output_pdfs:
            if os.path.exists(pdf):
                print(f"  - {pdf}")


if __name__ == "__main__":
    main()
