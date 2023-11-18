# Import the modules
import numpy as np
import matplotlib.pyplot as plt
import pickle
from DDPG import DDPG
from sklearn.ensemble import GradientBoostingRegressor
from env import Environment
import numpy as np
import scipy.stats as stats

# Define the parameters
MAX_EPISODES = 1000 # The maximum number of episodes for training the DDPG agent
MAX_EP_STEPS = 200 # The maximum number of steps for each episode
REWARD_FACTOR = 1.0 # The factor to scale the reward
PENALTY_FACTOR = 1.0 # The factor to scale the penalty

# Create an instance of the DDPG agent
agent = DDPG(state_dim=10, action_dim=5, action_bound=1.0, lr_a=0.001, lr_c=0.002, gamma=0.9, batch_size=32, memory_size=10000)
env = Environment(n_users=10, n_servers=5, bandwidth=100, latency=0.1, utility=lambda x: np.log(1 + x))

# Initialize an empty list for the reward history
reward_history = []

# Start the training loop
for i in range(MAX_EPISODES):

    # Reset the environment and get the initial state
    state = env.reset()

    # Initialize the episode reward
    ep_reward = 0

    # Start the episode loop
    for j in range(MAX_EP_STEPS):

        # Choose an action based on the state
        action = agent.choose_action(state)

        # Execute the action and get the next state, reward, and done flag
        next_state, reward, done, info = env.step(action)

        # Scale the reward and the penalty
        reward = reward * REWARD_FACTOR
        penalty = info['penalty'] * PENALTY_FACTOR

        # Add the reward and the penalty to the episode reward
        ep_reward += (reward - penalty)

        # Store the transition in the replay buffer
        agent.replay_buffer.append((state, action, reward, next_state, done))

        # Learn from the replay buffer
        agent.learn()

        # Update the state
        state = next_state

        # Check if the episode is done
        if done:
            break

    # Print the episode reward
    print('Episode: {}, Reward: {}'.format(i, ep_reward))

    # Append the episode reward to the reward history
    reward_history.append(ep_reward)

# Save the reward history to a file
np.save('reward.npy', reward_history)

# Save the DDPG agent to a file
agent.save('model.pkl')

# Plot the reward history
plt.plot(reward_history)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('DDPG Learning Curve')
plt.show()

# Access the replay buffer of the DDPG agent
replay_buffer = agent.replay_buffer

# Initialize an empty list for the dataset
dataset = []

# Iterate over the replay buffer
for state, action, reward, next_state, done in replay_buffer:

    # Append the state and action to the dataset
    dataset.append((state, action))

# Save the dataset to a file
with open("dataset.pkl", "wb") as f:
    pickle.dump(dataset, f)

# Load the dataset from the file
with open("dataset.pkl", "rb") as f:
    dataset = pickle.load(f)

# Split the dataset into features and labels
X = np.array([state for state, action in dataset]) # The features are the states
y = np.array([action for state, action in dataset]) # The labels are the actions

# Create an instance of the GBDT model
gbdt = GradientBoostingRegressor()

# Train the GBDT model on the dataset
gbdt.fit(X, y)

# Save the GBDT model to a file
with open("gbdt.pkl", "wb") as f:
    pickle.dump(gbdt, f)

# Load the GBDT model from the file
with open("gbdt.pkl", "rb") as f:
    gbdt = pickle.load(f)

# Evaluate the GBDT model on the resource allocation problem
# You can use any metric you want, such as the reward, the utility, or the QoS
# Here is an example of using the reward as the metric

# Initialize an empty list for the evaluation reward history
eval_reward_history = []

# Start the evaluation loop
for i in range(100):

    # Reset the environment and get the initial state
    state = env.reset()

    # Initialize the evaluation episode reward
    eval_ep_reward = 0

    # Start the evaluation episode loop
    for j in range(MAX_EP_STEPS):

        # Predict an action based on the state using the GBDT model
        action = gbdt.predict(state.reshape(1, -1))

        # Execute the action and get the next state, reward, and done flag
        next_state, reward, done, info = env.step(action)

        # Scale the reward and the penalty
        reward = reward * REWARD_FACTOR
        penalty = info['penalty'] * PENALTY_FACTOR

        # Add the reward and the penalty to the evaluation episode reward
        eval_ep_reward += (reward - penalty)

        # Update the state
        state = next_state

        # Check if the episode is done
        if done:
            break

    # Print the evaluation episode reward
    print('Evaluation Episode: {}, Reward: {}'.format(i, eval_ep_reward))

    # Append the evaluation episode reward to the evaluation reward history
    eval_reward_history.append(eval_ep_reward)

# Save the evaluation reward history to a file
np.save('eval_reward.npy', eval_reward_history)

# Plot the evaluation reward history
plt.plot(eval_reward_history)
plt.xlabel('Evaluation Episode')
plt.ylabel('Reward')
plt.title('GBDT Evaluation Curve')
plt.show()

# Load the reward history of the DDPG agent from the file
reward_history = np.load('reward.npy')

# Load the evaluation reward history of the GBDT model from the file
eval_reward_history = np.load('eval_reward.npy')

# Calculate the mean and the standard deviation of the reward for each model
ddpg_mean = np.mean(reward_history)
ddpg_std = np.std(reward_history)
gbdt_mean = np.mean(eval_reward_history)
gbdt_std = np.std(eval_reward_history)

# Print the mean and the standard deviation of the reward for each model
print('DDPG Mean Reward: {:.2f}, DDPG Standard Deviation: {:.2f}'.format(ddpg_mean, ddpg_std))
print('GBDT Mean Reward: {:.2f}, GBDT Standard Deviation: {:.2f}'.format(gbdt_mean, gbdt_std))

# Perform a t-test to compare the reward of the two models
t, p = stats.ttest_ind(reward_history, eval_reward_history)

# Print the t-statistic and the p-value
print('T-statistic: {:.2f}, P-value: {:.2f}'.format(t, p))

# Interpret the result
if p < 0.05:
    print('The difference in reward between the two models is statistically significant.')
else:
    print('The difference in reward between the two models is not statistically significant.')