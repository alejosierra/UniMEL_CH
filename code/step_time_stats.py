"""Compute training-step and evaluation-time statistics from a JSONL log."""

import json
import re
import statistics
from pathlib import Path


def parse_elapsed_time(value: str) -> float:
    """Convert a log duration such as ``1h 2m 5s`` to seconds."""
    units = {"h": 3600, "m": 60, "s": 1}
    return sum(
        float(amount) * units[unit]
        for amount, unit in re.findall(r"(\d+(?:\.\d+)?)\s*([hms])", value)
    )


def step_time_stats(log_path: str | Path) -> tuple[float, float]:
    records = []
    with Path(log_path).open() as log_file:
        for line in log_file:
            record = json.loads(line)
            if "global_step/max_steps" not in record or "elapsed_time" not in record:
                continue
            step = int(record["global_step/max_steps"].split("/", 1)[0])
            elapsed = parse_elapsed_time(record["elapsed_time"])
            records.append((step, elapsed))

    seconds_per_step = [
        (current_elapsed - previous_elapsed) / (current_step - previous_step)
        for (previous_step, previous_elapsed), (current_step, current_elapsed) in zip(
            records, records[1:]
        )
        if current_step > previous_step
    ]

    if not seconds_per_step:
        raise ValueError("The log does not contain two records with increasing steps")

    return statistics.mean(seconds_per_step), statistics.stdev(seconds_per_step)


def evaluation_time_stats(log_path: str | Path) -> tuple[float, float]:
    seconds_per_evaluation_step = []
    with Path(log_path).open() as log_file:
        for line in log_file:
            record = json.loads(line)
            if "eval_steps_per_second" in record:
                steps_per_second = float(record["eval_steps_per_second"])
                if steps_per_second > 0:
                    seconds_per_evaluation_step.append(1 / steps_per_second)

    if len(seconds_per_evaluation_step) < 2:
        raise ValueError("The log does not contain at least two evaluation step rates")

    return statistics.mean(seconds_per_evaluation_step), statistics.stdev(
        seconds_per_evaluation_step
    )


def evaluation_step_counts(log_path: str | Path) -> list[int]:
    counts = []
    with Path(log_path).open() as log_file:
        for line in log_file:
            record = json.loads(line)
            if "eval_runtime" in record and "eval_steps_per_second" in record:
                count = round(
                    float(record["eval_runtime"])
                    * float(record["eval_steps_per_second"])
                )
                counts.append(count)

    if not counts:
        raise ValueError("The log does not contain evaluation runtime and step-rate fields")

    return counts


if __name__ == "__main__":
    import sys

    step_mean, step_stddev = step_time_stats(sys.argv[1])
    eval_mean, eval_stddev = evaluation_time_stats(sys.argv[1])
    eval_counts = evaluation_step_counts(sys.argv[1])
    print(f"Training average: {step_mean:.4f} s/step")
    print(f"Training std. dev.: {step_stddev:.4f} s/step")
    print(f"Evaluation average: {eval_mean:.4f} s/step")
    print(f"Evaluation std. dev.: {eval_stddev:.4f} s/step")
    print(f"Evaluation steps per evaluation: {', '.join(map(str, eval_counts))}")