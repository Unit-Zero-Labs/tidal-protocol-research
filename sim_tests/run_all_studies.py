#!/usr/bin/env python3
"""
Master script to run all 10 studies sequentially.
Each study will save results to its own directory.

Estimated total runtime: 80-120 minutes
"""

import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def format_time(seconds):
    """Format seconds into human-readable time"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def run_study(study_num, module_name, description, expected_runtime_min):
    """Run a single study and track its performance"""
    print("\n" + "=" * 80)
    print(f"STUDY {study_num}: {description}")
    print("=" * 80)
    print(f"Expected runtime: ~{expected_runtime_min} minutes")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    start_time = time.time()
    
    try:
        # Import and run the study module
        module = __import__(f"sim_tests.{module_name}", fromlist=['main'])
        module.main()
        
        elapsed = time.time() - start_time
        
        print(f"\n✅ Study {study_num} completed successfully!")
        print(f"   Actual runtime: {format_time(elapsed)}")
        
        return {
            'study': study_num,
            'description': description,
            'status': 'SUCCESS',
            'runtime': elapsed,
            'expected': expected_runtime_min * 60
        }
        
    except Exception as e:
        elapsed = time.time() - start_time
        
        print(f"\n❌ Study {study_num} failed!")
        print(f"   Error: {str(e)}")
        print(f"   Runtime before failure: {format_time(elapsed)}")
        
        return {
            'study': study_num,
            'description': description,
            'status': 'FAILED',
            'runtime': elapsed,
            'expected': expected_runtime_min * 60,
            'error': str(e)
        }


def main():
    print("=" * 80)
    print("HIGH TIDE vs AAVE: 10-STUDY SIMULATION SUITE")
    print("=" * 80)
    print()
    print("This will run all 10 studies sequentially:")
    print("  - Studies 1-5: Symmetric (Historical rates for both)")
    print("  - Studies 6-10: Asymmetric (Advanced MOET for High Tide)")
    print()
    print("Estimated total runtime: 80-120 minutes")
    print("Results will be saved to: tidal_protocol_sim/results/")
    print()
    
    input("Press ENTER to start all studies, or Ctrl+C to cancel...")
    print()
    
    overall_start = time.time()
    
    # Define all studies
    studies = [
        # Symmetric Studies (1-5)
        (1, "run_study_1_2021_mixed_symmetric", "2021 Mixed Market (Symmetric)", 8),
        (2, "run_study_2_2024_bull_symmetric", "2024 Bull Market - Equal HF (Symmetric)", 8),
        (3, "run_study_3_2024_capital_efficiency_symmetric", "2024 Capital Efficiency (Symmetric)", 8),
        (4, "run_study_4_2022_bear_symmetric", "2022 Bear Market (Symmetric)", 8),
        (5, "run_study_5_2025_lowvol_symmetric", "2025 Low Vol (Symmetric)", 6),
        
        # Asymmetric Studies (6-10)
        (6, "run_study_6_2021_mixed_asymmetric", "2021 Mixed Market (Advanced MOET)", 12),
        (7, "run_study_7_2024_bull_asymmetric", "2024 Bull Market - Equal HF (Advanced MOET)", 12),
        (8, "run_study_8_2024_capital_efficiency_asymmetric", "2024 Capital Efficiency (Advanced MOET)", 12),
        (9, "run_study_9_2022_bear_asymmetric", "2022 Bear Market (Advanced MOET)", 12),
        (10, "run_study_10_2025_lowvol_asymmetric", "2025 Low Vol (Advanced MOET)", 10),
    ]
    
    results = []
    
    # Run each study
    for study_num, module_name, description, expected_min in studies:
        result = run_study(study_num, module_name, description, expected_min)
        results.append(result)
        
        # Calculate remaining time
        elapsed_so_far = time.time() - overall_start
        completed_count = study_num
        remaining_count = 10 - completed_count
        avg_time_per_study = elapsed_so_far / completed_count
        estimated_remaining = avg_time_per_study * remaining_count
        
        if remaining_count > 0:
            print(f"\n📊 Progress: {completed_count}/10 studies complete")
            print(f"   Estimated time remaining: {format_time(estimated_remaining)}")
            print(f"   Estimated completion: {(datetime.now() + timedelta(seconds=estimated_remaining)).strftime('%H:%M:%S')}")
    
    # Final summary
    overall_elapsed = time.time() - overall_start
    
    print("\n" + "=" * 80)
    print("ALL STUDIES COMPLETE")
    print("=" * 80)
    print(f"Total runtime: {format_time(overall_elapsed)}")
    print()
    
    # Summary table
    print("Study Summary:")
    print("-" * 80)
    print(f"{'Study':<8} {'Status':<10} {'Runtime':<15} {'Description':<45}")
    print("-" * 80)
    
    success_count = 0
    for result in results:
        status_symbol = "✅" if result['status'] == 'SUCCESS' else "❌"
        status = f"{status_symbol} {result['status']}"
        runtime = format_time(result['runtime'])
        
        print(f"{result['study']:<8} {status:<10} {runtime:<15} {result['description']:<45}")
        
        if result['status'] == 'SUCCESS':
            success_count += 1
    
    print("-" * 80)
    print(f"\nSuccess rate: {success_count}/10 studies")
    
    if success_count == 10:
        print("\n🎉 All studies completed successfully!")
        print("\nResults saved to: tidal_protocol_sim/results/")
        print("\nNext steps:")
        print("  1. Review comparison charts in each study's charts/ directory")
        print("  2. Analyze survival rates (especially Studies 4 & 9)")
        print("  3. Compare BTC accumulation across all studies")
        print("  4. Compare symmetric vs asymmetric performance")
    else:
        print(f"\n⚠️  {10 - success_count} study(ies) failed. Check errors above.")
        print("\nFailed studies:")
        for result in results:
            if result['status'] == 'FAILED':
                print(f"  - Study {result['study']}: {result.get('error', 'Unknown error')}")
    
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user. Exiting...")
        sys.exit(1)

