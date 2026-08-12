import json
import pandas as pd
import numpy as np

def analyze_reranking_performance(json_file_path):
    """
    Read a JSON file with model evaluation results from multiple runs,
    group metrics by 'before' and 'after' reranking, and compute mean and std.
    
    Args:
        json_file_path (str): Path to the JSON file containing the results
    """
    
    # Read the JSON file
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Identify metrics by their suffixes
    # We'll group metrics into 'before' and 'after' categories
    before_metrics = [col for col in df.columns if col.endswith('_before')]
    after_metrics = [col for col in df.columns if col.endswith('_after')]
    
    # Create separate DataFrames for before and after
    df_before = df[before_metrics]
    df_after = df[after_metrics]
    
    # Rename columns to remove '_before' and '_after' suffixes for cleaner output
    df_before.columns = [col.replace('_before', '') for col in before_metrics]
    df_after.columns = [col.replace('_after', '') for col in after_metrics]
    
    # Compute mean and std for each metric
    before_stats = df_before.agg(['mean', 'std']).round(4)
    after_stats = df_after.agg(['mean', 'std']).round(4)
    
    # Combine results into a single DataFrame
    results = pd.DataFrame({
        'before_mean': before_stats.loc['mean'],
        'before_std': before_stats.loc['std'],
        'after_mean': after_stats.loc['mean'],
        'after_std': after_stats.loc['std']
    })
    
    # Sort by metric name
    results = results.sort_index()
    
    # Print results in a formatted way
    print("\n" + "="*60)
    print("MODEL PERFORMANCE: BEFORE vs AFTER RERANKING")
    print("="*60)
    
    for metric in results.index:
        before_mean = results.loc[metric, 'before_mean']
        before_std = results.loc[metric, 'before_std']
        after_mean = results.loc[metric, 'after_mean']
        after_std = results.loc[metric, 'after_std']
        
        print(f"\n{metric.upper()}:")
        print(f"  Before: {before_mean:.4f} Â± {before_std:.4f}")
        print(f"  After:  {after_mean:.4f} Â± {after_std:.4f}")
        
        # Calculate improvement
        improvement = after_mean - before_mean
        if improvement > 0:
            print(f"  Improvement: +{improvement:.4f}")
        elif improvement < 0:
            print(f"  Degradation: {improvement:.4f}")
        else:
            print(f"  No change")
    
    # Save results to a CSV file
    results.to_csv('reranking_performance_summary.csv')
    print(f"\nSummary saved to 'reranking_performance_summary.csv'")
    
    return results

if __name__ == "__main__":
    # Path to your JSON file
    json_file = "results_combined.json"  # Change this to your actual file path
    
    try:
        results = analyze_reranking_performance(json_file)
        print(f"\nAnalysis complete. Results are available in the output.")
    except FileNotFoundError:
        print(f"Error: File '{json_file}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
