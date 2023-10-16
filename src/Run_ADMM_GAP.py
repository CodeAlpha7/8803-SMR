import scipy.io
import numpy as np

from functions import *
from parameters import *
from ADMM import admm_opt_algorithm
from ADMM import admm_td3_algorithm
from ADMM import admm_static_algorithm


if __name__ == "__main__":

    #utility = np.zeros(ADMM_iter)
    INDEX = np.arange(SliceNum)

    utility_static, gap_static = admm_static_algorithm(SliceNum, UENum, RESNum, alpha, weight)
    print("********** Utility Static *******")
    print(utility_static)

    utility, gap = admm_td3_algorithm(SliceNum, UENum, RESNum, alpha, weight, INDEX)
    print("********** Utility *******")

    print(utility)

    utility_opt, gap_opt = admm_opt_algorithm(SliceNum, UENum, RESNum, alpha, weight)
    print("********** Utility optimized *******")
    print(utility_opt)

    scipy.io.savemat('new_results/result_ADMM_GAP.mat', mdict={'utility': utility, 'utility_opt': utility_opt, 'utility_static': utility_static,
                                                                             'gap': gap, 'gap_opt': gap_opt, 'gap_static': gap_static,})

