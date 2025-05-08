Hello!

Below you can find a outline of how to reproduce my solution for the WiDS Datathon 2025 competition.
If you run into any trouble with the setup/code or have any questions please contact me at snow.cedar.yao@gmail.com

# HARDWARE:
AWS g4dn.16xlarge Instance

- Custom 2nd generation Intel Xeon Scalable (Cascade Lake) processors
- 64 virtual CPUs (vCPUs)
- 32 physical cores (with Intel Hyper-Threading Technology)
- 256 GiB of RAM
- NVIDIA T4 Tensor Core GPUs
- 4 NVIDIA T4 GPUs

# SOFTWARE and DATA SETUP:
See Dockerfile for software packages used

To build the container: `docker build -t competition .`

Download data and set data path, eg `datapath=/mnt/computational-bio-data/user-scratch/byao/widsdatathon2025`

To run the code insdie container: `docker run --gpus all -v $datapath:/data competition`

The output prediction `submission.csv` will be stored in data path. Other files in data path is read only. Another run will overwrite existing `submission.csv`.


# DATA PROCESSING
## Train and predict
- Option 1: train and predict. `docker run --rm --gpus all -v $datapath:/data competition`
- Option 2: interactive. `docker run -it --rm --gpus all -v $datapath:/data  --entrypoint /bin/bash competition`
  Once inside the container, `sh run.sh`
## Predict on a new test set
Edit paths in `config.SETTINGS.json`.

In the interactive mode, first run `sh run.sh`, then run `python3 -u predict_new.py`