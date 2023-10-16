import time
import pickle

import numpy as np
import torch
import matplotlib.pyplot as matplt

from parameters import *
from env_mra import ResourceEnv
# from ddpg_alg_spinup import ddpg
from td3_sb3 import td3
from ddpg_sb3 import ddpg


if __name__ == "__main__":

    with open("pickled_data/saved_alpha.pickle", "wb") as fileop:
        pickle.dump(alpha, fileop)

    with open("pickled_data/saved_weight.pickle", "wb") as fileop:
        pickle.dump(weight, fileop)

    ########################################################################################################################
    ##########################################        Main Training           #############################################
    ########################################################################################################################
    start_time = time.time()
    utility = np.zeros(SliceNum)
    x = np.zeros([UENum, maxTime], dtype=np.float32)

    for i in range(SliceNum):
        policy_kwargs = dict(net_arch=hidden_sizes, activation_fn=torch.nn.ReLU)

        logger_kwargs = dict(output_dir='td3_model/' + str(RESNum) + 'slice' + str(i), exp_name=str(RESNum) + 'slice_exp' + str(i))

        env = ResourceEnv(alpha=alpha[i], weight=weight[i],
                          num_res=RESNum, num_user=UENum,
                          max_time=maxTime, min_reward=minReward,
                          rho=rho, test_env=False)

        utility[i], _ = td3(env=env, policy_kwargs=policy_kwargs,
                            steps_per_epoch=steps_per_epoch,
                            epochs=epochs, lr=pi_lr,
                            start_steps=start_steps, batch_size=batch_size,
                            seed=seed, replay_size=replay_size, max_ep_len=maxTime,
                            logger_kwargs=logger_kwargs, fresh_learn_idx=True)
        

        # utility[i], _ = ddpg(env=env, policy_kwargs=policy_kwargs,
        #                     steps_per_epoch=steps_per_epoch,
        #                     epochs=epochs, lr=pi_lr,
        #                     start_steps=start_steps, batch_size=batch_size,
        #                     seed=seed, replay_size=replay_size, max_ep_len=maxTime,
        #                     logger_kwargs=logger_kwargs, fresh_learn_idx=True)

        print('slice' + str(i) + 'training completed.')

    end_time = time.time()
    print('Training Time is ' + str(end_time - start_time))

    #####################################          result ploting            ###############################################

    with open("pickled_data/saved_alpha.pickle", "rb") as fileop:
        load_alpha = pickle.load(fileop)
        print(load_alpha)

    with open("pickled_data/saved_weight.pickle", "rb") as fileop:
        load_weight = pickle.load(fileop)
        print(load_weight)

    # print(weight)

    # matplt.subplot(2, 1, 1)
    # matplt.plot(sum_utility)
    # matplt.subplot(2, 1, 2)
    # matplt.plot(sum_x)
    matplt.show()

    print('done')
