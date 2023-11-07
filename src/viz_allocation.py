import os
import re
import warnings
import argparse

import numpy as np
import matplotlib.pyplot as plt


TARGET_FILENAME = "Run_ADMM_GAP_trace.txt"


warnings.filterwarnings("ignore", category=UserWarning)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Visualize allocation trace of a model')
    parser.add_argument('model_dir', type=str, help='Directory of the model')
    return parser.parse_args()


def parse_matrix(allocation_block: str) -> list:
    allocation_matrix = []
    lines = allocation_block.split("\n")
    for idx, line in enumerate(lines):
        if '[' not in line:
            continue

        numbers = [float(x) for x in re.findall(r'\d+\.\d+|\d+', line)]
        allocation_matrix.append(numbers)
        if ']]' in line:
            break

    return allocation_matrix


def main():
    args = parse_args()
    model_dir = args.model_dir

    trace_file = f"{model_dir}/{TARGET_FILENAME}"

    if not os.path.exists(trace_file):
        print("\nModel directory does not have a allocation trace file")

        print("\nCreating allocation trace file...\n")

        td3 = "td3"
        ddpg = "ddpg"
        if td3 in model_dir:
            cmd = f"python src/Run_ADMM_GAP.py td3 \"{model_dir}\""

        elif ddpg in model_dir:
            cmd = f"python src/Run_ADMM_GAP.py ddpg \"{model_dir}\""

        print(f"\n> Running command: {cmd}\n")
        os.system(cmd)


    with open(trace_file, "r") as f:
        data = f.read()

    # Getting algo name
    algo = re.search(r"\*{9,}\s*(TD3|DDPG)\s*\*", data).group(1)

    # Getting Algo allocation 
    allocations = data.split("current allocation is")
    for allocation in allocations:
        if "**** Utility ****" in allocation:
            break

    algo_allocation = parse_matrix(allocation)

    # Getting ADMM allocation 
    allocations = data.split("current allocation is")
    for allocation in allocations:
        if "**** Utility optimized ****" in allocation:
            break

    admm_allocation = parse_matrix(allocation)


    N = len(algo_allocation)  # Number of rows
    M = len(algo_allocation[0])  # Number of columns

    fig, axes = plt.subplots(N, M, figsize=(12, 8), sharex='col', sharey='row')

    colors = ['b', 'g']

    for i in range(N):
        for j in range(M):
            ax = axes[i, j]
            x = np.arange(2)

            width = 0.6

            ax.bar(x, [algo_allocation[i][j], admm_allocation[i][j]], width, label=[algo, 'ADMM'], color=colors)
            ax.set_xticks(x)
            ax.set_xticklabels([algo, 'ADMM'])

            yticks = ax.get_yticks()
            ax.set_yticklabels([f'{int(y)}%' for y in yticks])

    for i in range(N):
        axes[i, 0].set_ylabel(f'Resource {i+1}', rotation=0, ha='right', va='center')

    for j in range(M):
        axes[0, j].set_title(f'Slice {j+1}', va='bottom')

    for ax in axes.flatten():
        ax.yaxis.grid(True, alpha=0.5)

    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    plt.savefig(f"{model_dir}/plots/allocation.png")


if __name__ == '__main__':
    main()
