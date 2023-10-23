import os
import argparse

import scipy.io
import numpy as np

from functions import *
from parameters import *
from ADMM import admm_opt_algorithm, admm_td3_algorithm, admm_static_algorithm, admm_ddpg_algorithm


DST_DIR = "new_results"


def parse_args():
    parser = argparse.ArgumentParser(description='ADMM Algorithm')
    parser.add_argument(
        'algorithm',
        choices=['td3', 'ddpg'],
        help="Choose the RL algorithm (td3 or ddpg)"
    )
    parser.add_argument('model_path', type=str, help='Path to the model to be used')
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = parse_args()

    model_name = args.model_path.split("/")[-2]

    #utility = np.zeros(ADMM_iter)
    INDEX = np.arange(SliceNum)

    utility_static, gap_static = admm_static_algorithm(SliceNum, UENum, RESNum, alpha, weight)
    print("********** Utility Static *******")
    print(utility_static)

    if args.algorithm == 'ddpg':
        utility, gap = admm_ddpg_algorithm(SliceNum, UENum, RESNum, alpha, weight, INDEX, model_path=args.model_path)

    elif args.algorithm == 'td3':
        utility, gap = admm_td3_algorithm(SliceNum, UENum, RESNum, alpha, weight, INDEX, model_path=args.model_path)

    print("********** Utility *******")

    print(utility)

    utility_opt, gap_opt = admm_opt_algorithm(SliceNum, UENum, RESNum, alpha, weight)
    print("********** Utility optimized *******")
    print(utility_opt)

    os.makedirs(DST_DIR, exist_ok=True)

    scipy.io.savemat(f'{DST_DIR}/result_ADMM_GAP_{model_name}.mat', mdict={'utility': utility, 'utility_opt': utility_opt, 'utility_static': utility_static,
                                                                             'gap': gap, 'gap_opt': gap_opt, 'gap_static': gap_static,})

