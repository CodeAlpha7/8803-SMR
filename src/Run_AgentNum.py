import scipy.io
import numpy as np

from functions import *
from parameters import *
from ADMM import admm_mix_algorithm


if __name__ == "__main__":

    simulated_optimization = [True] * SliceNum
    utility = np.zeros(SliceNum)

    for i in range(SliceNum):

        for j in range(i):

            simulated_optimization[j] = False

        utility[i] = admm_mix_algorithm(SliceNum, UENum, RESNum, alpha, weight, simulated_optimization)[-1]

    scipy.io.savemat('new_results/result_agent_num.mat', mdict={'utility': utility,})

