"""Quick setup script for gold price predictor enhancements.

Run this script to set up all enhanced features and train models.
"""

import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


def run_command(cmd: list[str], description: str) -> bool:
    """Run a command and return success status."""
    print(f"\n{'='*60}")
    print(f"[*] {description}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(
            [sys.executable, "-m"] + cmd,
            cwd=BASE_DIR,
            check=True,
            capture_output=False,
        )
        print(f"[OK] {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[!] Error in {description}: {e}")
        return False
    except KeyboardInterrupt:
        print(f"\n[!] {description} interrupted by user")
        return False


def main():
    """Main setup workflow."""
    print("="*60)
    print("Gold Price Predictor Enhancement Setup")
    print("="*60)
    print("\nThis script will:")
    print("  1. Fetch exogenous features (economic indicators)")
    print("  2. Fetch news sentiment data")
    print("  3. Create enhanced dataset")
    print("  4. Train enhanced models")
    print("\nNote: This may take 30-60 minutes depending on your data size.")
    print("      You can skip steps by commenting them out in the script.")
    
    response = input("\nContinue? (y/n): ").strip().lower()
    if response != "y":
        print("Setup cancelled.")
        return
    
    steps = [
        (["backend.services.fetch_exogenous"], "Fetching exogenous features"),
        (["backend.services.news_sentiment"], "Fetching news sentiment"),
        (["backend.services.preprocess_enhanced"], "Creating enhanced dataset"),
        (["backend.services.trainer_enhanced", "--models", "rf", "xgb", "ensemble"], "Training models"),
    ]
    
    success_count = 0
    for cmd, description in steps:
        if run_command(cmd, description):
            success_count += 1
        else:
            print(f"\n[!] Setup failed at: {description}")
            print("    You can continue manually or fix the issue and rerun.")
            response = input("    Continue anyway? (y/n): ").strip().lower()
            if response != "y":
                break
    
    print("\n" + "="*60)
    print(f"Setup complete! {success_count}/{len(steps)} steps successful.")
    print("="*60)
    print("\nNext steps:")
    print("  1. Check model_metrics.json for model performance")
    print("  2. Update your API to use enhanced models (see ENHANCEMENT_GUIDE.md)")
    print("  3. Test predictions with: python -m backend.services.predictor_enhanced")


if __name__ == "__main__":
    main()

