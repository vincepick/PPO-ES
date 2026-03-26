import numpy as np
import sys
import os
import json
import re
import argparse
from pathlib import Path
import matplotlib.pyplot as plt


DEFAULT_JSON_PATH = "/cs/home/vap2/Documents/diss/PPO-ES_GECCO24/bbob_optima.json"


def load_optima(json_path):
    with open(json_path, "r") as f:
        return json.load(f)


def get_f_opt(optima, dim, instance, problem_index):
    dim_key = f"dim_{dim}"
    inst_key = f"instance_{instance}"
    func_key = f"f{problem_index:02d}"
    return float(optima[dim_key][inst_key][func_key]["f_opt"])


def parse_filename(file_path):
    filename = os.path.basename(file_path)
    pattern = r"fitness_episode_(\d+)_problem_(\d+)_instance_(\d+)\.npy"
    match = re.match(pattern, filename)
    if not match:
        raise ValueError(f"Filename does not match expected pattern: {filename}")
    episode, problem, instance = map(int, match.groups())
    return episode, problem, instance


def get_dimension_from_path(file_path):
    current = Path(file_path).resolve().parent

    for parent in [current] + list(current.parents):
        match = re.match(r"DIM_(\d+)", parent.name)
        if match:
            return int(match.group(1))

    raise ValueError(
        f"Could not find a parent directory matching DIM_<dim> for file: {file_path}"
    )


def compute_auc_per_run(file_path, optima, clamp_below_fopt=False):
    data = np.load(file_path, allow_pickle=True)

    if not isinstance(data, (list, np.ndarray)):
        raise TypeError(f"Expected a list or array of lists in file: {file_path}")

    dim = get_dimension_from_path(file_path)
    _, problem, instance = parse_filename(file_path)
    f_opt = get_f_opt(optima, dim, instance, problem)

    aucs = []

    for i, run in enumerate(data):
        if not isinstance(run, (list, np.ndarray)) or len(run) == 0:
            print(f"Skipping invalid or empty run {i} in {file_path}")
            continue

        values = np.asarray(run, dtype=float)
        shifted = values - f_opt

        if clamp_below_fopt:
            shifted = np.maximum(shifted, 0.0)

        area = np.trapz(shifted)
        aucs.append(float(area))

    return problem, aucs


def find_matching_files(input_path, target_episode):
    input_path = Path(input_path)

    if input_path.is_file():
        files = [input_path]
    elif input_path.is_dir():
        files = sorted(input_path.rglob("fitness_episode_*_problem_*_instance_*.npy"))
    else:
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    matched = []
    for file_path in files:
        try:
            episode, _, _ = parse_filename(str(file_path))
            if episode == target_episode:
                matched.append(file_path)
        except ValueError:
            continue

    return matched


def plot_boxplot(problem_to_aucs, output_path, episode):
    boxplot_data = []
    labels = []

    for problem in range(1, 25):
        aucs = problem_to_aucs.get(problem, [])
        if len(aucs) == 0:
            boxplot_data.append([np.nan])
        else:
            boxplot_data.append(aucs)
        labels.append(str(problem))

    plt.figure(figsize=(14, 7))
    plt.boxplot(boxplot_data, labels=labels)
    plt.xlabel("Problem Function")

    plt.ylabel("Area Under Curve (log scale)")
    plt.yscale("log")

    plt.title(f"Box Plots For Each Problem Function For Episode: {episode}")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Plot a box-and-whisker plot of run AUCs for all problems at a given episode."
        )
    )
    parser.add_argument(
        "input_path",
        help="Path to a .npy file or a directory containing .npy files",
    )
    parser.add_argument(
        "output_file",
        help="Path to save the output plot, e.g. auc_boxplot.png",
    )
    parser.add_argument(
        "--episode",
        type=int,
        default=1200,
        help="Episode number to filter on. Default: 1200",
    )
    parser.add_argument(
        "--optima-json",
        default=DEFAULT_JSON_PATH,
        help=f"Path to bbob_optima.json. Default: {DEFAULT_JSON_PATH}",
    )
    parser.add_argument(
        "--clamp-below-fopt",
        action="store_true",
        help="Clamp shifted values below zero before integrating.",
    )

    args = parser.parse_args()

    optima = load_optima(args.optima_json)
    matched_files = find_matching_files(args.input_path, args.episode)

    if not matched_files:
        raise ValueError(
            f"No matching .npy files found for episode {args.episode} in {args.input_path}"
        )

    problem_to_aucs = {problem: [] for problem in range(1, 25)}

    for file_path in matched_files:
        try:
            problem, aucs = compute_auc_per_run(
                str(file_path),
                optima,
                clamp_below_fopt=args.clamp_below_fopt,
            )
            problem_to_aucs[problem].extend(aucs)
            print(
                f"Processed {file_path} | problem={problem} | runs_added={len(aucs)}"
            )
        except Exception as e:
            print(f"Skipping {file_path} due to error: {e}")

    for problem in range(1, 25):
        print(f"Problem {problem}: {len(problem_to_aucs[problem])} run AUC values")

    plot_boxplot(problem_to_aucs, args.output_file, args.episode)
    print(f"Saved plot to: {args.output_file}")


if __name__ == "__main__":
    main()