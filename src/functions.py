import numpy as np
import cvxpy as cp
import time

from parameters import *
from env_mra import ResourceEnv
from stable_baselines3 import DDPG


def run_policy(env, model, max_ep_len=None, num_episodes=100, render=True):
    ep_action, ep_reward, ep_utility = [], [], []
    o, info = env.reset()
    r, d, ep_ret, ep_len, n = 0, False, 0, 0, 0
    while n < num_episodes:
        if render:
            env.render()
            time.sleep(1e-3)
        a, _ = model.predict(o)
        o, r, d, _, info = env.step(a)
        a = (a + 1) / 2
        ep_ret += r
        ep_len += 1
        ep_action.append(np.reshape(a, [env.num_res, env.UENum]))
        ep_reward.append(r)
        ep_utility.append(info["utility"])
        if d or (ep_len == max_ep_len):
            print('Episode %d \t EpRet %.3f \t EpLen %d'%(n, ep_ret, ep_len))
            o, info = env.reset()
            r, d, ep_ret, ep_len, n = 0, False, 0, 0, 0
            n += 1

    return np.array(ep_reward), np.array(ep_action), np.array(ep_utility)


def load_and_run_policy(agent_id, alpha, weight, UENum, RESNum, aug_penalty, model_path):
    slice_path = f"{model_path}/{RESNum}slice{agent_id}.zip"
    model = DDPG.load(slice_path)
    env = ResourceEnv(
        alpha=alpha,
        weight=weight,
        num_user=UENum,
        num_res=RESNum,
        min_reward=minReward,
        max_time=maxTime,
        rho=rho,
        aug_penalty=aug_penalty,
        test_env=True
    )
    ep_reward, ep_action, ep_utility = run_policy(env, model, num_episodes=1, render=False)
    return np.sum(ep_reward), ep_action, np.sum(ep_utility)


def simple_convex_alogrithm(z_minus_u, alpha, weight, UENum, minReward=0.0):

    # Do not use numpy when you use cvxopt!!!!!!!!!!@@@@################################################################
    x = cp.Variable(UENum)

    y = [weight[i] * (x[i]**float(alpha[i]))/alpha[i] for i in range(UENum)]

    const = [(x[i]**float(alpha[i]))/alpha[i] for i in range(UENum)]  # the requirement only for real reward (not weighted)

    if use_other_utility_function:
        y = [weight[i] * RESNum/(RESNum * cp.exp(- alpha[i] * x[i]) + 1) for i in range(UENum)]
        const = [ RESNum/(RESNum * cp.exp(- alpha[i] * x[i])+ 1) for i in range(UENum)]

    fx = cp.sum(y) - 0.5 * rho * cp.sum_squares(cp.sum(x) - z_minus_u)

    objective = cp.Minimize(-fx)

    factor, iter, maxiter = 1, 1, 10  # decrease the factor, i.e., loose the constraint, when we cannot solve the problem optimally

    # solve the problem, if not optimal or not well solved, reduce the constraint and do it again
    while True:

        constraints = [factor*minReward <= const[i] for i in range(UENum)] + [cp.sum(x) <= Rmax]

        prob = cp.Problem(objective, constraints,)

        # The optimal objective value is returned by `prob.solve()`.
        result = prob.solve() #gp=True

        assert(iter <= maxiter)

        if prob.status == 'optimal':
            break
        else:
            factor *= 0.5
            iter += 1

    optimal_x = x.value

    utility, real_utility, const_ = np.zeros(UENum), np.zeros(UENum), np.zeros(UENum)

    for i in range(UENum):
        const_[i] = (optimal_x[i]**alpha[i])/alpha[i]
        utility[i] = weight[i] * (optimal_x[i]**alpha[i])/alpha[i] - maxTime * np.clip(minReward - const_[i], 0, None)
        real_utility[i] = weight[i] * (optimal_x[i]**alpha[i])/alpha[i]

        if use_other_utility_function:
            const_[i] = Rmax/(Rmax * np.exp(- alpha[i] * optimal_x[i])+ 1)
            utility[i] = weight[i] * Rmax/(Rmax * np.exp(- alpha[i] * optimal_x[i])+ 1) - maxTime * np.clip(minReward - const_[i], 0, None)
            real_utility[i] = weight[i] * Rmax/(Rmax * np.exp(- alpha[i] * optimal_x[i])+ 1)

    aug_penalty = 0.5 * rho * np.abs(np.sum(optimal_x) - z_minus_u)

    # The optimal value for x is stored in `x.value`.
    return np.sum(utility)-aug_penalty, optimal_x, np.sum(real_utility)


def simple_static_alogrithm(z_minus_u, alpha, weight, UENum, minReward=0.0):

    optimal_x = z_minus_u / UENum * np.ones(UENum)

    utility, real_utility, const_ = np.zeros(UENum), np.zeros(UENum), np.zeros(UENum)
    for i in range(UENum):
        const_[i] = (optimal_x[i]**alpha[i])/alpha[i]
        utility[i] = weight[i] * (optimal_x[i]**alpha[i])/alpha[i] - maxTime * np.clip(minReward - const_[i], 0, None)
        real_utility[i] = weight[i] * (optimal_x[i]**alpha[i])/alpha[i]
        if use_other_utility_function:
            const_[i] = Rmax/(Rmax * np.exp(- alpha[i] * optimal_x[i])+ 1)
            utility[i] = weight[i] * Rmax/(Rmax * np.exp(- alpha[i] * optimal_x[i])+ 1) - maxTime * np.clip(minReward - const_[i], 0, None)
            real_utility[i] = weight[i] * Rmax/(Rmax * np.exp(- alpha[i] * optimal_x[i])+ 1)

    aug_penalty = 0.5 * rho * np.abs(np.sum(optimal_x) - z_minus_u)

    # The optimal value for x is stored in `x.value`.
    return np.sum(utility)-aug_penalty, optimal_x, np.sum(real_utility)

def optimize_z_function(sumx, u,SliceNum):

    z = cp.Variable(SliceNum)

    objective = cp.Minimize(cp.sum_squares(sumx -z + u))

    constraints = [0 <= z, cp.sum(z) == Rmax]

    prob = cp.Problem(objective, constraints)

    result = prob.solve()

    return z.value

