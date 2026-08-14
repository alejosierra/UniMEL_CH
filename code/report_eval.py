import argparse
import json
import re
import pandas as pd


def analyze_reranking_performance(json_file_path):
    """
    Read a JSON file with model evaluation results from multiple runs,
    group metrics by 'before' and 'after' reranking, and compute mean and std.

    Metric names may look like "recall_before@1", "recall_after@10",
    or "MAP_before". The phase marker can appear anywhere in the name,
    not only at the end.

    Args:
        json_file_path (str): Path to the JSON file containing the results
    """

    # Read the JSON file
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Group columns by metric name so the output can be indexed by phase.
    metric_columns = {}
    pattern = re.compile(r"^(?P<metric>.*)_(?P<phase>before|after)(?P<suffix>.*)$")

    for column in df.columns:
        match = pattern.match(column)
        if match is None:
            continue

        metric_name = match.group('metric') + match.group('suffix')
        phase = match.group('phase')
        metric_columns.setdefault(metric_name, {})[phase] = column

    phases = ['before', 'after']
    rows = {}
    for phase in phases:
        phase_stats = {}
        for metric_name, columns in metric_columns.items():
            if phase not in columns:
                continue
            series = df[columns[phase]]
            phase_stats[(metric_name, 'mean')] = round(series.mean(), 4)
            phase_stats[(metric_name, 'std')] = round(series.std(), 4)
        rows[phase] = phase_stats

    results = pd.DataFrame.from_dict(rows, orient='index')
    results.index.name = 'phase'
    if not results.empty:
        results.columns = pd.MultiIndex.from_tuples(results.columns, names=['metric', 'stat'])
        results = results.sort_index(axis=1, level=[0, 1])

    # multiply all metrics by 100 to convert to percentage
    results *= 100

    # Print results in a formatted way
    print("\n" + "="*60)
    print("MODEL PERFORMANCE: BEFORE vs AFTER RERANKING")
    print("="*60)

    for metric in results.columns.get_level_values('metric').unique():
        before_mean = results.loc['before', (metric, 'mean')]
        before_std = results.loc['before', (metric, 'std')]
        after_mean = results.loc['after', (metric, 'mean')]
        after_std = results.loc['after', (metric, 'std')]

        print(f"\n{metric.upper()}:")
        print(f"  Before: {before_mean:.4f} +/- {before_std:.4f}")
        print(f"  After:  {after_mean:.4f} +/- {after_std:.4f}")

        improvement = after_mean - before_mean
        if improvement > 0:
            print(f"  Improvement: +{improvement:.4f}")
        elif improvement < 0:
            print(f"  Degradation: {improvement:.4f}")
        else:
            print("  No change")

    # reorder columns to get the metrics in this order: recall@k (sorted by k), MAP, MRR
    def metric_sort_key(metric):
        if metric.startswith('recall@'):
            k = int(metric.split('@')[1])
            return (0, k)
        elif metric == 'MAP':
            return (1, 0)
        elif metric == 'MRR':
            return (2, 0)
        else:
            return (3, 0)  # Other metrics come last

    results = results.reindex(sorted(results.columns, key=lambda x: metric_sort_key(x[0])), axis=1)

    # Save results to a CSV file
    csv_filename = json_file_path.replace('.json', '_summary.csv')
    results.to_csv(csv_filename)
    print(f"\nSummary saved to '{csv_filename}'")

    return results

if __name__ == "__main__":
    # Path to your JSON file
    args = argparse.ArgumentParser(description="Analyze reranking performance from JSON results.")
    args.add_argument('json_file', type=str, help='Path to the JSON file containing evaluation results.')
    parsed_args = args.parse_args()
    json_file = parsed_args.json_file

    try:
        results = analyze_reranking_performance(json_file)
        print(f"\nAnalysis complete. Results are available in the output.")
    except FileNotFoundError:
        print(f"Error: File '{json_file}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
