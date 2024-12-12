# Final Code for CS598: How to do Research (Group 3)

This repository contains code to reproduce the experiments of our work with Improved Intra-Operator Parallelism for Distributed LLM
Training.

## Install

### Necessary Hardware
Currently this codebase only works on MacOS because the shell file is designed for MacOS.

To run this project, you need to install the required packages. Follow the steps below to install the dependencies using the [requirements.txt](requirements.txt) file.

1. Clone the repository: 
```bash
git clone https://github.com/livingshade/Metis.git
```

2. Navigate to the project directory:
```bash 
cd Metis
```

3. Install dependencies using the requirements.txt file: 
```bash
# If using conda
conda create -n HTDR python=3.11 pip -y
conda activate HTDR

pip install -r requirements.txt
```
## Usage (First Row of Table 1)
1. Once all dependencies are installed, you are ready to start running experiments. First, generate synthetic profiling data for a 10 layer model with a global batch size of 128.
```bash
python3 gen_synth_data.py 10 128 
```

2. Create your hostfile. To make things easier, we provided a script that can quickly generate a cluster made up of a 50/50 split of A100 and V100s nodes. Each node has 4 GPU cards. 
```bash
python3 gen_hostfile.py 16
```

3. Run the simulation experiment with naive Metis. Please make sure to pass in the absolute file path on your device to Metis (the local repository) to `HOME_DIR`. The results should be found in `logs/GPT_1.5B.log`. If the count value is `612`, then this step was done correctly.
```bash
sh ./scripts/mac_cost_het_cluster.sh HOME_DIR='ABSOLUTE_FILE_PATH_TO_METIS' MODEL_NAME=GPT MODEL_SIZE=1.5B NUM_LAYERS=10 GBS=128  MAX_PROFILED_TP=128 MAX_PROFILED_BATCH_SIZE=128 SCALE_VARIANCE=1 MAX_PERMUTE_LEN=128 TRIALS=10 USE_STRAT=False
```

4. Run the simulation experiment with our method (same commands as before but change `USE_STRAT` to `True`). Please make sure to pass in the absolute file path on your device to Metis (the local repository) to `HOME_DIR`. The results should be found in `logs/GPT_1.5B.log`. If the count value is `562`, then this step was done correctly.
```bash
sh ./scripts/mac_cost_het_cluster.sh HOME_DIR='ABSOLUTE_FILE_PATH_TO_METIS' MODEL_NAME=GPT MODEL_SIZE=1.5B NUM_LAYERS=10 GBS=128  MAX_PROFILED_TP=128 MAX_PROFILED_BATCH_SIZE=128 SCALE_VARIANCE=1 MAX_PERMUTE_LEN=128 TRIALS=10 USE_STRAT=True
```

## How to read results
Count is the number of search steps, average time is the average wall clock time out of `TRIALS` trials. The second value in the 4th line of `logs/GPT_1.5B.log` is the cost of the best plan.

## Recreating Rest of the Table
NOTE: Wall-clock time has variance due to many factors such as hardware and current load on your host device. We mitigated this variance by taking the average of 10 trials (set by the `TRIALS=10`) argument. Cost of best plan of our method may be occasionally worse than Metis, but on average, the costs of our plans is no worse than Metis's. The step counts are fixed and our count should ALWAYS be lower than Metis's.

### Second Row of Table 1 (20 layers, 32 A100s + 32 V100s)
Same exact commands as previous row, but change the 10 to a 20 in the first command, 16 to 32 in the second, and 10 to 20 for `NUM_LAYERS` when running either shell script. Please make sure to pass in the absolute file path on your device to Metis (the local repository) to `HOME_DIR`. This may take a while. The results should be found in `logs/GPT_1.5B.log`. If the count value of Metis is `58078` and ours is `41208`, then this step was done correctly.
```bash
python3 gen_synth_data.py 20 128 
python3 gen_hostfile.py 32

# Use Metis
sh ./scripts/mac_cost_het_cluster.sh HOME_DIR='ABSOLUTE_FILE_PATH_TO_METIS' MODEL_NAME=GPT MODEL_SIZE=1.5B NUM_LAYERS=20 GBS=128  MAX_PROFILED_TP=128 MAX_PROFILED_BATCH_SIZE=128 SCALE_VARIANCE=1 MAX_PERMUTE_LEN=128 TRIALS=5 USE_STRAT=False

# Use ours
sh ./scripts/mac_cost_het_cluster.sh HOME_DIR='ABSOLUTE_FILE_PATH_TO_METIS' MODEL_NAME=GPT MODEL_SIZE=1.5B NUM_LAYERS=20 GBS=128  MAX_PROFILED_TP=128 MAX_PROFILED_BATCH_SIZE=128 SCALE_VARIANCE=1 MAX_PERMUTE_LEN=128 TRIALS=5 USE_STRAT=True
```

### Third Row of Table 1 (20 layers, 64 A100s + 64 V100s)
Same exact commands as previous row, but 32 to 64 in the second command. Please make sure to pass in the absolute file path on your device to Metis (the local repository) to `HOME_DIR`. The results should be found in `logs/GPT_1.5B.log`. If the count value of Metis is `3734` and ours is `3482`, then this step was done correctly.
```bash
python3 gen_synth_data.py 20 128 
python3 gen_hostfile.py 64

# Use Metis
sh ./scripts/mac_cost_het_cluster.sh HOME_DIR='ABSOLUTE_FILE_PATH_TO_METIS' MODEL_NAME=GPT MODEL_SIZE=1.5B NUM_LAYERS=20 GBS=128 MAX_PROFILED_TP=128 MAX_PROFILED_BATCH_SIZE=128 SCALE_VARIANCE=1 MAX_PERMUTE_LEN=128 TRIALS=5 USE_STRAT=False

# Use ours
sh ./scripts/mac_cost_het_cluster.sh HOME_DIR='ABSOLUTE_FILE_PATH_TO_METIS' MODEL_NAME=GPT MODEL_SIZE=1.5B NUM_LAYERS=20 GBS=128 MAX_PROFILED_TP=128 MAX_PROFILED_BATCH_SIZE=128 SCALE_VARIANCE=1 MAX_PERMUTE_LEN=128 TRIALS=5 USE_STRAT=True
```

### Fourth Row Row of Table 1 (40 layers, 128 A100s + 128 V100s)
This one takes a very long time (30 min per trial), so we only use 1 trial. Same exact commands as previous row, but change the 20 to a 40 in the first command, 64 to 128 in the second, and 20 to 40 for `NUM_LAYERS` when running either shell script. Please make sure to pass in the absolute file path on your device to Metis (the local repository) to `HOME_DIR`. The results should be found in `logs/GPT_1.5B.log`. If the count value of Metis is `1714619` and ours is `1553492`, then this step was done correctly.
```bash
python3 gen_synth_data.py 40 128 
python3 gen_hostfile.py 128

# Use Metis
sh ./scripts/mac_cost_het_cluster.sh HOME_DIR='ABSOLUTE_FILE_PATH_TO_METIS' MODEL_NAME=GPT MODEL_SIZE=1.5B NUM_LAYERS=40 GBS=128 MAX_PROFILED_TP=128 MAX_PROFILED_BATCH_SIZE=128 SCALE_VARIANCE=1 MAX_PERMUTE_LEN=128 TRIALS=1 USE_STRAT=False

# Use ours
sh ./scripts/mac_cost_het_cluster.sh HOME_DIR='ABSOLUTE_FILE_PATH_TO_METIS' MODEL_NAME=GPT MODEL_SIZE=1.5B NUM_LAYERS=40 GBS=128 MAX_PROFILED_TP=128 MAX_PROFILED_BATCH_SIZE=128 SCALE_VARIANCE=1 MAX_PERMUTE_LEN=128 TRIALS=1 USE_STRAT=True
```

#### Supported Python Versions
- 3.11

## Training Experiments and Supplementary Graphs
The two graphs found in the paper were generated in [graphs.ipynb](./graphs.ipynb). They are found in [plots](./plots/).

#### Hardware
2 [c240g5](https://docs.cloudlab.us/hardware.html) Cloudlab nodes, each fitted with one P100 GPU.
Simulation was done on a 2021 Apple M1 Macbook Pro.