# Profiling

This dir contains profiling results.

## How to profile a script
    
```bash 
python -m cProfile -o <output_file> <script.py> <args>
```

e.g.
```bash 
python -m cProfile -o prof/train_all_slices_DDPG_vm.prof src/train_all_slices.py ddpg
```

In this case the profiling results will be saved in the file `prof/train_all_slices_DDPG_vm.prof`.

## How to visualize the profiling results
To read and visualize the profiling results we are using [snakeviz](https://jiffyclub.github.io/snakeviz/).

```bash
snakeviz <output_file>
```

e.g.
```bash
snakeviz prof/train_all_slices_DDPG_vm.prof
```

This will start a server and open a new tab in your browser where the profiling results are served.
