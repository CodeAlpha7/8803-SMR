# Distillation using GBDT

UNDERSTANDING THE CHANGES:
1. DDPG.py
it has a single get_status function which sets the current status on the memory. 
when the algorithm has been run, it stores transitions in the memory buffer. So, a self_pointer is used to keep track of how many of these transitions are stored.
So, we loop through the entire memory to gather all this transition information and store it in a "transition" dictionary.
we store STATE, ACTION, REWARD, NEXT STATE
sets key so that it is accessible in the returned status.

2. close_policy.py
USER_NUM = 2 (originally 10)
EDGE_NUM = 2 (originally 10)
SCREEN_RENDER = false (originally true)

3. env.py
same as close_policy.py

4. render.py
same: Edge_size = 3 (20), user_size = 3 (10)
these are the hyperparameters

5. run_this.py
stores transitions to CSV file. these transitions are:
EPISODE, OFFLOAD, BANDWIDTH, ACTION, REWARDS, NEXT_BANDWIDTH and NEXT_OFFLOAD

training loop has maximum number of episodes defined by LEARNING_MAX_EPISODE



------------------------------------------------------------
IMPLEMENTATION IDEA:
------------------------------------------------------------
1. Collect training data during the DDPG training loop. Collect state, action and reward data for each episode.
2. Preprocess the data - transformations into appropriate feature vectors and labels suitable to be used for training.
3. Split, train and evaluate.



------------------------------------------------------------
PERFORMANCE METRICS:
------------------------------------------------------------
1. Mean Squared Error (MSE): to measure difference between predicted values and true values (in dataset). A measure of average squared deviation - useful in quantifying the overall accuracy or goodness of fit of a model.

2. R-squared error: proportion of variance in the dependent variable (target) whichis explained by the independent variables (features). 
 - captures model's ability to account for variability in the data. We want a larger R-squared value. 
 - value is between 0 and 1. this value reflects the percentage of variance explained. So, we want 1 = perfectly explains all variance.
 - higher value indicates that the model captures more of the underlying patterns in data


MSE = measures the magnitude of errors
R-squared = proportion of variance explained

BOTH ARE COMPLIMENTARTY. COMBINING BOTH HELPS GIVE A COMPREHENSIVE VIEW OF MODEL'S PERFORMANCE.



3. state space versus execution time
 - generate a set of state space samples - representing different states that the models need to process.
 - for each sample, measure the execution time required by both.
 - compare and visualize.



-----------------------------------------------------------------
features = ['offload', 'bandwidth', 'action', 'next_bandwidth', 'next_offload']

Here, feature importance is only given to ACTION, BANDWIDTH and OFFLOAD
highest importance by a big margin to ACTION

mean squared error = 0.22
ACTION values between 0 and 2


------------------------------------------------------------------

DDPG model:
does an action to transition from one state to another
provides reward - feedback for chosen action

so, information in each CSV file:
1. State information = offload, bandwidth, action, nextBW, nextOF
2. Action
3. Reward
4. Next State


GBDT model
Output = reward value for a given set of features

predicted rewards vs actual reward



-----------------------------------------------------------------
STATE SPACE vs EXECUTION TIME:
Take state as input and measure the time taken to make predictions based on that state. Do this for both DDPG and GBDT, compare them.

so, generate state space samples - a set of data points that represent a state or configuration of env. SPECIFIC INSTANCES.

Basically, we want to measure that time it takes to go from:
State A to State B when an action is performed.
so, state space = 5, 10, 15 
means the time taken for predicting these many samples.

