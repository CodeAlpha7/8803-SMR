import time

import numpy as np
from numpy.typing import DTypeLike
import matplotlib.pyplot as matplt

from stable_baselines3 import DDPG
from stable_baselines3.common.noise import ActionNoise

class DecayingNoise(ActionNoise):
    def __init__(self, mean: np.ndarray, sigma: np.ndarray, total_steps: int, dtype: DTypeLike = np.float32) -> None:
        self._mu = mean
        self._sigma = sigma
        self.decay_ratio = np.exp(np.log(1e-3)/total_steps)
        self._dtype = dtype
        super().__init__()

    def __call__(self) -> np.ndarray:
        rand = np.random.normal(self._mu, self._sigma).astype(self._dtype)
        self._sigma *= self.decay_ratio
        return rand

    def __repr__(self) -> str:
        return f"DecayingNoise(mu={self._mu}, sigma={self._sigma})"



def ddpg(env, policy_kwargs=dict(), seed=0,
        steps_per_epoch=5000, epochs=100, replay_size=int(1e6), gamma=0.99,
        polyak=0.995, lr=1e-3, batch_size=1000, start_steps=10000,
        update_after=1000, update_every=50, act_noise=1.0, target_noise=0.2,
        noise_clip=0.5, policy_delay=2, num_test_episodes=10, max_ep_len=100,
        logger_kwargs=dict(), save_freq=1, fresh_learn_idx=True):
    """
    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: A function which takes in placeholder symbols
            for state, ``x_ph``, and action, ``a_ph``, and returns the main
            outputs from the agent's Tensorflow computation graph:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``pi``       (batch, act_dim)  | Deterministically computes actions
                                           | from policy given states.
            ``q``        (batch,)          | Gives the current estimate of Q* for
                                           | states in ``x_ph`` and actions in
                                           | ``a_ph``.
            ``q_pi``     (batch,)          | Gives the composition of ``q`` and
                                           | ``pi`` for states in ``x_ph``:
                                           | q(x, pi(x)).
            ===========  ================  ======================================

        ac_kwargs (dict): Any kwargs appropriate for the actor_critic
            function you provided to DDPG.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs)
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs to run and train agent.

        replay_size (int): Maximum length of replay buffer.

        gamma (float): Discount factor. (Always between 0 and 1.)

        polyak (float): Interpolation factor in polyak averaging for target
            networks. Target networks are updated towards main networks
            according to:

            .. math:: \\theta_{\\text{targ}} \\leftarrow
                \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta

            where :math:`\\rho` is polyak. (Always between 0 and 1, usually
            close to 1.)

        pi_lr (float): Learning rate for policy.

        q_lr (float): Learning rate for Q-networks.

        batch_size (int): Minibatch size for SGD.

        start_steps (int): Number of steps for uniform-random action selection,
            before running real policy. Helps exploration.

        act_noise (float): Stddev for Gaussian exploration noise added to
            policy at training time. (At test time, no noise is added.)

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.
        fresh_learn_idx (boolean): Whether this time is fresh/first learn
            if True, we collect the data from true env and save the results,
            if False, we try to load the results(prev saved) and do no-interact learning
    """
    
    # The noise objects for DDPG
    n_actions = env.action_space.shape[-1]
    total_steps = steps_per_epoch * epochs
    action_noise = DecayingNoise(mean=np.zeros(n_actions), sigma=np.ones(n_actions), total_steps=total_steps)
    model = DDPG("MlpPolicy", env, action_noise=action_noise, batch_size=batch_size, learning_starts=start_steps, learning_rate=lr, gamma=gamma, seed=seed, verbose=1, policy_kwargs=policy_kwargs)

    test_env = env

    def test_agent():
        for j in range(num_test_episodes):
            o, info = test_env.reset()
            d, ep_ret, ep_len, ep_act = False, 0, 0, []
            while not (d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time (noise_scale=0)
                a, _states = model.predict(o)
                o, r, d, _, _ = test_env.step(a)
                ep_ret += r
                ep_len += 1
                ep_act.append(a)
        return ep_ret, ep_act

    start_time = time.time()
    o, info = env.reset()
    ep_ret = 0 
    ep_len0 = 0
    total_steps = steps_per_epoch * epochs
    total_reward, total_action = [], []
    decay_ratio = np.exp(np.log(1e-3)/total_steps)

    # Main loop: collect experience in env and update/log each epoch   
    model.learn(total_timesteps=total_steps, log_interval=10)
    model.save(logger_kwargs["output_dir"])

    # Test the performance of the deterministic version of the agent.
    test_ret, test_act = test_agent()
    total_reward.append(test_ret)
    total_action.append(test_act)

    # matplt.figure()
    # matplt.plot(total_reward)
    # matplt.savefig('slice_training' + str(int(time.time())) + '.png')
    # matplt.show()

    return total_reward[-1], np.transpose(total_action[-1])  # for reshape the results
