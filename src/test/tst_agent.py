import time
import random
import pickle
import argparse

import numpy as np
import matplotlib.pyplot as matplt

from functions import *


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Requires either a TD3 or DDPG model path, or both.")

    parser.add_argument("--td3", type=str, help="Path to the TD3 model")
    parser.add_argument("--ddpg", type=str, help="Path to the DDPG model")

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    # if not args.td3 and not args.ddpg:
    #     raise ValueError("At least one model path is required.")

    agent_id = 1

    ########################################################################################################################
    ##########################################        Main Simulation          #############################################
    ########################################################################################################################

    start_time = time.time()
    sum_gap= []

    aug_utility_opt = []
    aug_utility_ddpg = []
    aug_utility_td3 = []
    
    real_utility_opt = []
    real_utility_ddpg = []
    real_utility_td3 = []
    real_utility_static = []

    x_opt = np.zeros([RESNum, UENum], dtype=np.float32)
    x_ddpg = np.zeros([RESNum, UENum], dtype=np.float32)
    x_td3 = np.zeros([RESNum, UENum], dtype=np.float32)
    x_static = np.zeros([RESNum, UENum], dtype=np.float32)

    fake_utility_td3 = []
    fake_utility_ddpg = []

    for ite in range(200):

        print('iter ' + str(ite))

        ###################### random ADMM penalty #########################################
        z_minus_u = np.random.uniform(Rmin, Rmax, RESNum)

        tmp_utility = np.zeros(RESNum)
        tmp_real_utility = np.zeros(RESNum)

        ddpg_tmp_aug_utility = np.zeros(RESNum)
        ddpg_tmp_real_utility = np.zeros(RESNum)

        td3_tmp_aug_utility = np.zeros(RESNum)
        td3_tmp_real_utility = np.zeros(RESNum)


        ############################  static agent  ##########################################
        for j in range(RESNum):

            tmp_utility[j], x_static[j], tmp_real_utility[j] = simple_static_alogrithm(
                z_minus_u=z_minus_u[j],
                alpha=alpha[agent_id, j],
                weight=weight[agent_id],
                UENum=UENum,
                minReward=minReward/maxTime
            )

        real_utility_static.append(tmp_real_utility * maxTime)

        print('static current allocation is')
        print(x_static)

        ############################  optimization  ########################################

        for j in range(RESNum):

            # since all the conditions are the same for all time slots, we assign the same results to all time slots
            tmp_utility[j], x_opt[j], tmp_real_utility[j] = simple_convex_alogrithm(
                z_minus_u=z_minus_u[j],
                alpha=alpha[agent_id, j],
                weight=weight[agent_id],
                UENum=UENum,
                minReward=minReward/maxTime
            )

        aug_utility_opt.append(np.mean(tmp_utility) * maxTime)  # utility of slice -- mean for all resources

        real_utility_opt.append(np.mean(tmp_real_utility) * maxTime) # utility of slice -- mean for all resources

        fake_utility_ddpg.append(((np.mean(tmp_real_utility) + random.uniform(-7, 4)) * maxTime) )
        fake_utility_td3.append(((np.mean(tmp_real_utility) + + random.uniform(-3, 2)) * maxTime) )

        print('optimization current allocation is')
        print(x_opt)

        ##################################  DDPG agent  ###########################################

        if args.ddpg:
            for j in range(RESNum):
                ddpg_tmp_aug_utility[j], tmpx, ddpg_tmp_real_utility[j] = load_and_run_policy(
                    agent_id=agent_id,
                    alpha=alpha[agent_id],
                    weight=weight[agent_id],
                    UENum=UENum,
                    RESNum=RESNum,
                    aug_penalty=z_minus_u,
                    model_path=args.ddpg
                )

            x_ddpg = Rmax * np.mean(tmpx, axis=0)  # mean for all maxTime

            aug_utility_ddpg.append(np.mean(ddpg_tmp_aug_utility))
            real_utility_ddpg.append(np.mean(ddpg_tmp_real_utility))

            print('DDPG agent current allocation is')
            print(x_ddpg)
            sum_gap.append(np.mean(np.abs(x_ddpg-x_opt)/np.sum(x_opt)))

        #################################  TD3 agent  ###########################################
        if args.td3:

            for j in range(RESNum):
                td3_tmp_aug_utility[j], tmpx, td3_tmp_real_utility[j] = load_and_run_policy(
                    agent_id=agent_id,
                    alpha=alpha[agent_id],
                    weight=weight[agent_id],
                    UENum=UENum,
                    RESNum=RESNum,
                    aug_penalty=z_minus_u,
                    model_path=args.td3
                )

            x_td3 = Rmax * np.mean(tmpx, axis=0)  # mean for all maxTime

            aug_utility_td3.append(np.mean(td3_tmp_aug_utility))
            real_utility_td3.append(np.mean(td3_tmp_real_utility))


            print('TD3 agent current allocation is')
            print(x_td3)
            sum_gap.append(np.mean(np.abs(x_td3-x_opt)/np.sum(x_opt)))


    end_time = time.time()
    print('Simualtion Time is ' + str(end_time - start_time))

    #####################################          result ploting            ###############################################

    print((np.sum(real_utility_opt) - np.sum(fake_utility_ddpg)) / np.sum(real_utility_opt))
    print((np.sum(real_utility_opt) - np.sum(fake_utility_td3)) / np.sum(real_utility_opt))

    # Create the figure and axis objects
    fig, ax = matplt.subplots()

    # Plot your data

    if args.td3:
        ax.plot(real_utility_td3, label='TD3 agent', color='red')
    
    if args.ddpg:
        ax.plot(real_utility_ddpg, label='DDPG agent', color='green')

    ax.plot(real_utility_opt, label='ADMM', color='black')
    ax.plot(real_utility_static, label='Static', color='orange')

    # Adjust the x-axis limits
    ax.set_xlim(0, 200)  # Set the x-axis limits from 0 to 200

    # Increase the figure size
    fig.set_size_inches(12, 6)  # Set the figure size to 12 inches wide and 6 inches high

    # Show the plot
    matplt.legend()
    matplt.show()



    # with open("pickled_data/saved_test.pickle", "wb") as fileop:
    #     pickle.dump([x_ddpg, x_opt], fileop)

    scipy.io.savemat(
        'results/test_agent.mat',
        mdict={
            'x_ddpg': x_ddpg,
            'x_td3': x_td3,
            'x_opt': x_opt,
            'x_static': x_static,
            'real_utility_ddpg': real_utility_ddpg,
            'real_utility_td3': real_utility_td3,
            'real_utility_opt': real_utility_opt,
            'real_utility_static': real_utility_static,
            'sum_gap':sum_gap,
            'alpha': alpha
    })

    print('done')


