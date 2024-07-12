# This script is adapted from https://github.com/siboehm/SGEMM_CUDA/blob/master/plot_benchmark_results.py

import re
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import argparse

matplotlib.style.use("fivethirtyeight")
matplotlib.style.use("seaborn-v0_8-talk")
matplotlib.rcParams["font.family"] = "monospace"
matplotlib.rcParams["figure.dpi"] = 200
plt.rcParams["savefig.facecolor"] = "white"


def parse_file(file):

    with open(file, "r") as f:
        lines = [line.strip() for line in f.readlines()]

    data = {"size": [], "kernel": [], "bandwidth": []}

    array_size_pattern = "Test (.*?): n = (.*)"
    kernel_name_pattern = "Kernel: "
    bandwidth_pattern = "AVG Latency = (.*?) ms, AVG Bandwidth = (.*?) GB/s"

    size = -1
    kernel_name = ""
    bandwidth = 0.0

    for line in lines:

        r = re.match(array_size_pattern, line)
        if r:
            size = int(r.group(2))

        r = re.match(kernel_name_pattern, line)
        if r:
            kernel_name = line[8:]

        r = re.match(bandwidth_pattern, line)
        if r:
            bandwidth = float(r.group(2))
            data["size"].append(size)
            data["kernel"].append(kernel_name)
            data["bandwidth"].append(bandwidth)

    return data


def plot(df, output_path):
    """
    The dataframe has 3 columns: size, kernel, bandwidth

    We want to plot the bandwidth for each kernel, for each size as a single seaborn multi-line plot.
    """

    plt.figure(figsize=(18, 10))
    colors = sn.color_palette("husl", len(df["kernel"].unique()))
    sn.lineplot(data=df, x="size", y="bandwidth", hue="kernel", palette=colors)
    # also plot points, but without legend
    sn.scatterplot(
        data=df, x="size", y="bandwidth", hue="kernel", palette=colors, legend=False
    )

    # set ticks at actual sizes
    plt.xticks(df["size"].unique())
    # rotate xticks, and align them
    plt.xticks(rotation=45, ha="right", rotation_mode="anchor")
    # add small lines at the xticks

    # display the kernel names right next to the corresponding line
    for i, kernel in enumerate(df["kernel"].unique()):
        # right align the text
        plt.text(
            df[df["kernel"] == kernel]["size"].iloc[-1],
            df[df["kernel"] == kernel]["bandwidth"].iloc[-1],
            f"{i}:{kernel}",
            color=colors[i],
            horizontalalignment="left",
            weight="medium",
        )

    # turn off the legend
    plt.gca().get_legend().remove()

    plt.title("Performance of different kernels")
    plt.xlabel("Array Size")
    plt.ylabel("Bandwidth (GB/s)")
    plt.tight_layout()

    plt.savefig(output_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log_path", default="./result.log", help="The path to log of profiling."
    )
    parser.add_argument(
        "--plot_path",
        default="./benchmark_results.png",
        help="The path to the output plot",
    )
    args = parser.parse_args()

    data = []
    results_dict = parse_file(args.log_path)

    df = pd.DataFrame(results_dict)
    plot(df, args.plot_path)
