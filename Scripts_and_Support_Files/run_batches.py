#!/usr/bin/env python3
"""
Batch processing script for sentiment analysis.
This script automatically processes the entire dataset in configurable batches.
"""

import subprocess
import sys
import time
import os
from datetime import datetime

def run_batch(start_row: int, end_row: int, api_key: str = None, model: str = "gpt-4"):
    """Run sentiment analysis for a specific batch."""
    cmd = [
        sys.executable, "ImpactAPIcalls.py",
        "--start-row", str(start_row),
        "--end-row", str(end_row),
        "--model", model
    ]
    
    if api_key:
        cmd.extend(["--api-key", api_key])
    
    print(f"\n🚀 Processing batch: rows {start_row} to {end_row}")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"✅ Batch {start_row}-{end_row} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Batch {start_row}-{end_row} failed with error code {e.returncode}")
        print(f"Error output: {e.stderr}")
        return False

def main():
    """Main function to run batch processing."""
    print("🔄 Sentiment Analysis Batch Processor")
    print("=" * 50)
    
    # Configuration
    BATCH_SIZE = 100  # Process 100 respondents at a time
    TOTAL_RESPONDENTS = 1310  # Total number of respondents
    
    # Get API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ OPENAI_API_KEY environment variable not set!")
        print("Please set your API key before running this script.")
        return
    
    # Get model choice
    print("Choose OpenAI model:")
    print("1. gpt-4 (more accurate, more expensive)")
    print("2. gpt-5 (slightly less expensive, may be less accurate with subtlety)")
    
    while True:
        choice = input("Enter choice (1 or 2): ").strip()
        if choice == "1":
            model = "gpt-4o"
            break
        elif choice == "2":
            model = "gpt-5"
            break
        else:
            print("Please enter 1 or 2")
    
    # Choose row range
    default_start = 1
    default_end = TOTAL_RESPONDENTS
    start_row_input = input(f"Start row (1-indexed, default {default_start}): ").strip()
    end_row_input = input(f"End row (1-indexed, default {default_end}): ").strip()

    try:
        start_row = int(start_row_input) if start_row_input else default_start
    except ValueError:
        start_row = default_start
    try:
        end_row = int(end_row_input) if end_row_input else default_end
    except ValueError:
        end_row = default_end

    # Clamp to valid bounds
    start_row = max(1, start_row)
    end_row = min(TOTAL_RESPONDENTS, end_row)
    if start_row > end_row:
        start_row, end_row = end_row, start_row

    # Calculate batches within chosen range
    batches = []
    for start in range(start_row, end_row + 1, BATCH_SIZE):
        end = min(start + BATCH_SIZE - 1, end_row)
        batches.append((start, end))
    
    print(f"\n📊 Processing rows {start_row} to {end_row} in {len(batches)} batches")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Model: {model}")
    
    # Confirm before starting
    response = input("\nProceed with batch processing? (y/N): ").strip().lower()
    if response != 'y':
        print("Batch processing cancelled.")
        return
    
    # Process batches
    successful_batches = 0
    failed_batches = []
    
    start_time = time.time()
    
    for i, (start_row, end_row) in enumerate(batches, 1):
        print(f"\n📦 Batch {i}/{len(batches)}")
        
        success = run_batch(start_row, end_row, api_key, model)
        
        if success:
            successful_batches += 1
        else:
            failed_batches.append((start_row, end_row))
        
        # Progress update
        print(f"Progress: {i}/{len(batches)} batches completed")
        print(f"Success rate: {successful_batches}/{i} ({successful_batches/i*100:.1f}%)")
        
        # Add delay between batches to avoid overwhelming the API
        if i < len(batches):
            print("⏳ Waiting 5 seconds before next batch...")
            time.sleep(5)
    
    # Summary
    end_time = time.time()
    total_time = end_time - start_time
    
    print("\n" + "=" * 50)
    print("🎯 BATCH PROCESSING COMPLETE")
    print("=" * 50)
    print(f"Total batches: {len(batches)}")
    print(f"Successful: {successful_batches}")
    print(f"Failed: {len(failed_batches)}")
    print(f"Success rate: {successful_batches/len(batches)*100:.1f}%")
    print(f"Total time: {total_time/60:.1f} minutes")
    
    if failed_batches:
        print(f"\n❌ Failed batches:")
        for start, end in failed_batches:
            print(f"   Rows {start} to {end}")
        print("\nYou can re-run failed batches individually using:")
        for start, end in failed_batches:
            print(f"   python ImpactAPIcalls.py --start-row {start} --end-row {end}")
    
    print(f"\n✅ All successful batches have been saved to CSV files")
    print("Check the current directory for output files.")

if __name__ == "__main__":
    main()
