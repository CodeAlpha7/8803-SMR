import os
import re
import argparse

import matplotlib.pyplot as plt


patterns = {
    "slice":            r"Start slice [0-9]+ training\.\.\.",
    "ep_rew_mean":      r"\|\s+ep_rew_mean\s+\|\s+([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s+\|",
    "episodes":         r"\|\s+episodes\s+\|\s+(\d+)\s+\|",
    "fps":              r"\|\s+fps\s+\|\s+(\d+)\s+\|",
    "time_elapsed":     r"\|\s+time_elapsed\s+\|\s+(\d+)\s+\|",
    "total_timesteps":  r"\|\s+total_timesteps\s+\|\s+(\d+)\s+\|",
    "actor_loss":       r"\|\s+actor_loss\s+\|\s+([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s+\|",
    "critic_loss":      r"\|\s+critic_loss\s+\|\s+([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s+\|",
    "n_updates":        r"\|\s+n_updates\s+\|\s+(\d+)\s+\|",
}


class SliceTrainingTrace:
    def __init__(self, slice_id: int):
        self.slice_id = slice_id
        self.ep_rew_mean = []
        self.episodes = []
        self.fps = []
        self.time_elapsed = []
        self.total_timesteps = []
        self.actor_loss = []
        self.critic_loss = []
        self.n_updates = []

    def ingest_raw_trace_table(self, raw_trace_table: str):
        self.ep_rew_mean.append(float(re.findall(patterns["ep_rew_mean"], raw_trace_table)[0]))
        self.episodes.append(int(re.findall(patterns["episodes"], raw_trace_table)[0]))
        self.fps.append(int(re.findall(patterns["fps"], raw_trace_table)[0]))
        self.time_elapsed.append(int(re.findall(patterns["time_elapsed"], raw_trace_table)[0]))
        self.total_timesteps.append(int(re.findall(patterns["total_timesteps"], raw_trace_table)[0]))
        self.actor_loss.append(float(re.findall(patterns["actor_loss"], raw_trace_table)[0]))
        self.critic_loss.append(float(re.findall(patterns["critic_loss"], raw_trace_table)[0]))
        self.n_updates.append(int(re.findall(patterns["n_updates"], raw_trace_table)[0]))

    def print(self):
        print(f"Slice {self.slice_id}")
        print(f"ep_rew_mean: {self.ep_rew_mean}")
        print(f"episodes: {self.episodes}")
        print(f"fps: {self.fps}")
        print(f"time_elapsed: {self.time_elapsed}")
        print(f"total_timesteps: {self.total_timesteps}")
        print(f"actor_loss: {self.actor_loss}")
        print(f"critic_loss: {self.critic_loss}")
        print(f"n_updates: {self.n_updates}")
        

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Visualize training trace of a model')
    parser.add_argument('model_dir', type=str, help='Directory of the model')
    return parser.parse_args()


def main():
    args = parse_args()
    model_dir = args.model_dir

    trace_file = f"{model_dir}/training_trace.txt"

    if not os.path.exists(trace_file):
        print("Model directory does not have a training trace file")
        print(trace_file)
        exit(1)

    with open(trace_file, "r") as f:
        data = f.read()

    slice_raw_traces = re.split(patterns["slice"], data)[1:]

    slice_traces = []

    # Looping over trace slices wise
    for i, slice_raw_trace in enumerate(slice_raw_traces):
        tables = slice_raw_trace.split("rollout/")[1:]

        slice_trace = SliceTrainingTrace(i)

        # Looping over trace tables wise
        for table in tables:
            slice_trace.ingest_raw_trace_table(table)

        slice_traces.append(slice_trace)

    # Plotting
    attributes_to_draw = ["ep_rew_mean", "fps", "actor_loss", "critic_loss"]
    os.makedirs(f"{model_dir}/plots", exist_ok=True)

    for attribute in attributes_to_draw:
        plt.figure()
        for slice_trace in slice_traces:
            plt.plot(slice_trace.episodes, getattr(slice_trace, attribute), label=f"Slice {slice_trace.slice_id}")
        plt.title(attribute)

        plt.xlabel('Episodes')
        plt.ylabel(attribute)
        plt.grid(True)

        plt.legend()
        plt.savefig(f"{model_dir}/plots/{attribute}.png")
        


if __name__ == '__main__':
    main()
