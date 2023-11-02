import argparse
import time
import pickle
import datetime
import os
import threading


import numpy as np
import torch
import matplotlib.pyplot as matplt

from parameters import *
from env_mra import ResourceEnv
from td3_sb3 import td3
from ddpg_sb3 import ddpg

from utils import redirect_output_to_file_and_stdout, reset_output


MODELS_DIR = "models"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train using TD3 or DDPG algorithms.")
    parser.add_argument(
        'algorithm',
        choices=['td3', 'ddpg'],
        help="Choose the RL algorithm (td3 or ddpg)"
    )
    parser.add_argument(
        '-n', '--name',
        type=str,
        default='',
        help="Custom name for the experiment"
    )
    args = parser.parse_args()
    return args


def train_slice(env, policy_kwargs, steps_per_epoch, epochs, lr, start_steps, batch_size, seed, replay_size, max_ep_len, logger_kwargs, fresh_learn_idx, result_list, index):
    if args.algorithm == 'td3':
        utility, _ = td3(env=env, policy_kwargs=policy_kwargs,
                         steps_per_epoch=steps_per_epoch,
                         epochs=epochs, lr=lr,
                         start_steps=start_steps, batch_size=batch_size,
                         seed=seed, replay_size=replay_size, max_ep_len=max_ep_len,
                         logger_kwargs=logger_kwargs, fresh_learn_idx=fresh_learn_idx)
    elif args.algorithm == 'ddpg':
        utility, _ = ddpg(env=env, policy_kwargs=policy_kwargs,
                          steps_per_epoch=steps_per_epoch,
                          epochs=epochs, lr=lr,
                          start_steps=start_steps, batch_size=batch_size,
                          seed=seed, replay_size=replay_size, max_ep_len=max_ep_len,
                          logger_kwargs=logger_kwargs, fresh_learn_idx=fresh_learn_idx)
    
    result_list[index] = utility


if __name__ == "__main__":

    args = parse_args()

    with open("pickled_data/saved_alpha.pickle", "wb") as fileop:
        pickle.dump(alpha, fileop)

    with open("pickled_data/saved_weight.pickle", "wb") as fileop:
        pickle.dump(weight, fileop)

    # Creating unique directory for the experiment
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    dst_dir_path = f"{MODELS_DIR}/{args.algorithm}_{timestamp}"

    # Adding custom name to the experiment if provided
    if args.name:
        dst_dir_path += f"_{args.name}"

    # Store desired parameters as text file
    os.makedirs(dst_dir_path, exist_ok=True)
    parameters_list = []

    parameters_list.extend([
        "ENVIRONMENT",
            ("RESNum", RESNum),  
            ("UENum", UENum),
            ("maxTime", maxTime),
            ("minReward", minReward),
            ("rho", rho),
            ("alpha", alpha),
            ("weight", weight),

        "ALGORITHM",
            ("SliceNum", SliceNum),
            ("seed", seed),
            ("hidden_sizes", hidden_sizes),
            ("replay_size", replay_size),
            ("epochs", epochs),
            ("steps_per_epoch", steps_per_epoch),
            ("batch_size", batch_size),
            ("pi_lr", pi_lr),
            ("start_steps", start_steps)
    ])

    with open(f"{dst_dir_path}/parameters.txt", "w") as fileop:
        for entry in parameters_list:

            if type(entry) == tuple:
                param_name, param_value = entry

                # if value is a long string, add custom formatting
                if str(param_value).count("\n") > 0:
                    param_value = str(param_value).replace("\n\n", "\n")
                    string_to_write = f"{param_name}:\n{param_value}\n\n"
                else:
                    string_to_write = f"{param_name}: {param_value}\n"

            else:
                string_to_write = f"{'-'*80}\n\t{entry}\n{'-'*80}\n"

            fileop.write(string_to_write)

    trace_file = redirect_output_to_file_and_stdout(f"{dst_dir_path}/training_trace.txt")

    ########################################################################################################################
    ##########################################        Main Training           #############################################
    ########################################################################################################################
    start_time = time.time()
    utility = np.zeros(SliceNum)
    x = np.zeros([UENum, maxTime], dtype=np.float32)

    results = [None] * SliceNum

    threads = []
    for i in range(SliceNum):
        print(f"Start slice {i} training...")

        policy_kwargs = dict(net_arch=hidden_sizes, activation_fn=torch.nn.ReLU)

        path = f"{dst_dir_path}/{RESNum}slice{i}"
        logger_kwargs = dict(output_dir=path, exp_name=str(RESNum) + 'slice_exp' + str(i))

        env = ResourceEnv(alpha=alpha[i], weight=weight[i],
                          num_res=RESNum, num_user=UENum,
                          max_time=maxTime, min_reward=minReward,
                          rho=rho, test_env=False)
        
        thread = threading.Thread(target=train_slice, args=(env, policy_kwargs, steps_per_epoch, epochs, pi_lr, start_steps, batch_size, seed, replay_size, maxTime, logger_kwargs, True, results, i))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    end_time = time.time()
    print('Training Time is ' + str(end_time - start_time))

    print(f"\nUtility:\n{results}\n\n")

    end_time = time.time()
    print('Training Time is ' + str(end_time - start_time))

    print(f"\nUtility:\n{utility}\n\n")

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
    # matplt.show()

    print('done')

    reset_output(trace_file)
