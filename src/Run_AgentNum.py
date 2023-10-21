import argparse

import scipy.io
import numpy as np

from functions import *
from parameters import *
from ADMM import admm_mix_algorithm


def parse_args():
    parser = argparse.ArgumentParser(description='ADMM Algorithm')
    parser.add_argument('model_path', type=str, help='Path to the model to be used')
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = parse_args()

    simulated_optimization = [True] * SliceNum
    utility = np.zeros(SliceNum)

    for i in range(SliceNum):

        for j in range(i):

            simulated_optimization[j] = False

        utility[i] = admm_mix_algorithm(SliceNum, UENum, RESNum, alpha, weight, simulated_optimization, model_path=args.model_path)[-1]

    scipy.io.savemat('new_results/result_agent_num.mat', mdict={'utility': utility,})

