# RARE
Source Codes for: Resource Allocation for network slicing using Reinforcement Learning

If you want to reset state information during new simulation run, try reinitiating your own saved_buffer. Try with empty buffer for new results.

## Setup
This setup is for Linux and Mac OS.

For Windows, [this link](https://docs.python.org/3/library/venv.html) might be useful.
### Requirements
- Python 3.6
- Conda

### Create a virtual environment
```bash
conda create --name cs8803 python=3.6 --file requirements_conda.yml
```
TODO: add conda reqs

### Activate the virtual environment
```bash
conda activate cs8803
```
and to deactivate:
```bash
conda deactivate
```

### Install dependencies
```bash
pip install -r requirements_pip.txt
```
Some dependencies require additional setup:
- OpenAI Spinup ([setup](https://spinningup.openai.com/en/latest/user/installation.html))
    - `cd spinningup`
    - `pip install -e .` or `conda develop .`

## Usage

# Refactoring TODOs

## Must haves
- [ ] Re-organize codebase structure
- [ ] Change absolute paths to relative paths
- [✓] Refactor import statements (built-in, third-party, local)
- [✓] Create environment
- [✓] List dependencies in `requirements.txt`
- [✓] Add usage and other instructions on README.md
- [ ] Run profiler
- [ ] Fix eventual bottlenecks

## Nice to haves
- [ ] Add some feedback when scrips run so user knows what is happening
