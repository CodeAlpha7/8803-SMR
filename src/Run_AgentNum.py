import time
import pickle

import scipy.io
import numpy as np
import cvxpy as cp
import tensorflow as tf
import matplotlib.pyplot as matplt

from utils import *
from tst_ddpg import *
from functions import *
from parameters import *
from env_mra import ResourceEnv
from ddpg_alg_spinup import ddpg
from ADMM import admm_mix_algorithm


if __name__ == "__main__":

    simulated_optimization = [True] * SliceNum
    utility = np.zeros(SliceNum)

    for i in range(SliceNum):

        for j in range(i):

            simulated_optimization[j] = False

        utility[i] = admm_mix_algorithm(SliceNum, UENum, RESNum, alpha, weight, simulated_optimization)[-1]

    scipy.io.savemat('results/result_agent_num.mat', mdict={'utility': utility,})

