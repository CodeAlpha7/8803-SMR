# RARE
Source Codes for: Resource Allocation for network slicing using Reinforcement Learning

If you want to reset state information during new simulation run, try reinitiating your own saved_buffer. Try with empty buffer for new results.

## Setup
This setup is for Linux and Mac OS.
For Windows, [this link](https://docs.python.org/3/library/venv.html) might be useful.

### Requirements
- Conda

### Create a virtual environment with conda requirements
```bash
conda create --name cs8803 python=3.9.18
```

### Activate the virtual environment
```bash
conda activate cs8803
```
and to deactivate:
```bash
conda deactivate
```

### Install additional PIP dependencies
```bash
pip install -r requirements.txt
```


**If using Spinup** some dependencies require additional setup:
- OpenAI Spinup ([setup](https://spinningup.openai.com/en/latest/user/installation.html))
    - Spinup module is already included in this repo:
    - `cd spinningup`
    - `pip install -e .` or `conda develop .`

# Usage
## Train slices
```bash
python src/train_all_slices.py [td3 | ddpg]
```
e.g.
```bash
python src/train_all_slices.py td3
```

Run `python src/train_all_slices.py -h` for more options.

You can also add a custom experiment name with `--name <name>`.
E.g.
```bash
python src/train_all_slices.py td3 --name my_experiment
```

### Results

Once the training is done, the results will be saved in `models/` and will have the following naming convention:

`<algo>_<timestamp>` if the name is not specified, or

`<algo>_<timestamp>_<name>` if the name is specified.

Inside such folder you will also find a `parameters.txt` file that lists the parameters used.

## Changes 

### TD3 Changes
- Previously the noise comes from a random number between 0 and 1, 
now I'm using the default in Stable Baselines 3 which is a Gaussian distribution
- Previously, the Replay Buffer class was defined by us,
I'm using the default from Stable Baselines 3
- Previously, the pi network had tanh as its activation function and the q networks used relu,
I haven't found a way to specify a different activation funciton for each so I believe they all use relu now
- I removed the logging done by SpinningUp, Stable Baselines 3 has a logger we can replace it with
Previously, there were a number of other network parameters we were able to define but now it doesn't look like we can,
such as polyak, target noise, noise clipping, waiting X steps to train while collecting samples
- Currently, all training is done in one call to learn() so we need to add periodic eval stages to it to log performance like we did previously instead of being able to call test_agent() in a loop
