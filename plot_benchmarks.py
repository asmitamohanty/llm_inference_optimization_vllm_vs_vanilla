import os
import argparse

import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (7, 5)
plt.rcParams["font.size"] = 12
plt.rcParams["axes.grid"] = True
plt.rcParams["savefig.dpi"] = 300


###########################################################
# Utility
###########################################################

def save_plot(output_file):
    plt.tight_layout()
    plt.savefig(output_file, bbox_inches="tight")
    plt.close()


###########################################################
# Context Sweep
###########################################################

def plot_context_metric(csvs, labels, metric, ylabel, output_dir):

    plt.figure()

    for csv, label in zip(csvs, labels):

        df = pd.read_csv(csv)

        plt.plot(
            df["context_length"],
            df[metric],
            marker="o",
            linewidth=2,
            label=label,
        )

    plt.xscale("log", base=2)

    plt.xlabel("Context Length (tokens)")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} vs Context Length")

    plt.legend()

    save_plot(
        os.path.join(
            output_dir,
            f"context_{metric}.png",
        )
    )


###########################################################
# Request Sweep
###########################################################

def plot_request_metric(csvs, labels, metric, ylabel, output_dir):

    plt.figure()

    for csv, label in zip(csvs, labels):

        df = pd.read_csv(csv)

        plt.plot(
            df["requests"],
            df[metric],
            marker="o",
            linewidth=2,
            label=label,
        )

    plt.xlabel("Concurrent Requests")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} vs Concurrent Requests")

    plt.legend()

    save_plot(
        os.path.join(
            output_dir,
            f"request_{metric}.png",
        )
    )


###########################################################
# Mixed Context
###########################################################

def plot_mixed_metric(csvs, labels, metric, ylabel, output_dir):

    values = []

    for csv in csvs:

        df = pd.read_csv(csv)

        values.append(df[metric].iloc[0])

    plt.figure()

    plt.bar(labels, values)

    plt.ylabel(ylabel)
    plt.title(f"Mixed Context {ylabel}")

    save_plot(
        os.path.join(
            output_dir,
            f"mixed_{metric}.png",
        )
    )


###########################################################
# Main
###########################################################
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--experiment",
        choices=[
            "context",
            "request",
            "mixed",
        ],
        required=True,
    )

    parser.add_argument(
        "--csv",
        nargs="+",
        required=True,
        help="CSV files to compare",
    )

    parser.add_argument(
        "--labels",
        nargs="+",
        required=True,
        help="Legend labels",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print("csv files:", args.csv)
    print("labels:", args.labels)
    if len(args.csv) != len(args.labels):
        raise ValueError(
            "--csv and --labels must have same length."
        )

    if args.experiment == "context":

        plot_context_metric(
            args.csv,
            args.labels,
            "latency_sec",
            "Latency (sec)",
            args.output_dir,
        )

        plot_context_metric(
            args.csv,
            args.labels,
            "ttft_sec",
            "TTFT (sec)",
            args.output_dir,
        )

        plot_context_metric(
            args.csv,
            args.labels,
            "tokens_per_sec",
            "Decode TPS",
            args.output_dir,
        )

    elif args.experiment == "request":

        plot_request_metric(
            args.csv,
            args.labels,
            "aggr_tps",
            "Aggregate TPS",
            args.output_dir,
        )

        plot_request_metric(
            args.csv,
            args.labels,
            "avg_latency_sec",
            "Average Latency (sec)",
            args.output_dir,
        )

        plot_request_metric(
            args.csv,
            args.labels,
            "p99_latency_sec",
            "P99 Latency (sec)",
            args.output_dir,
        )

        plot_request_metric(
            args.csv,
            args.labels,
            "avg_ttft_sec",
            "Average TTFT (sec)",
            args.output_dir,
        )

        plot_request_metric(
            args.csv,
            args.labels,
            "avg_itl_sec",
            "Average ITL (sec)",
            args.output_dir,
        )

    else:

        plot_mixed_metric(
            args.csv,
            args.labels,
            "avg_latency_sec",
            "Average Latency (sec)",
            args.output_dir,
        )

        plot_mixed_metric(
            args.csv,
            args.labels,
            "avg_ttft_sec",
            "Average TTFT (sec)",
            args.output_dir,
        )

        plot_mixed_metric(
            args.csv,
            args.labels,
            "avg_tps",
            "Average Decode TPS",
            args.output_dir,
        )

        plot_mixed_metric(
            args.csv,
            args.labels,
            "avg_itl_sec",
            "Average ITL (sec)",
            args.output_dir,
        )

        plot_mixed_metric(
            args.csv,
            args.labels,
            "p99_latency_sec",
            "P99 Latency (sec)",
            args.output_dir,
        )


if __name__ == "__main__":
    main()
