# RHiDE-ET

This folder contains the RHiDE-ET implementation. RHiDE stands for **Resource-adaptive Hierarchical Distributed Learning on the Edge**.

RHiDE uses split federated learning across IoT devices and edge servers. The model is divided at a selected layer: the device runs the first part and the edge server runs the remaining layers. If a device cannot train with its available resources, its raw data can be sent to the edge for processing.

**RHiDE-ET optimizes each training round independently.** It selects the split point, decides which devices participate, and considers raw-data transfer while keeping resource constraints feasible. It considers device and edge energy, device and edge memory, communication, and execution time.

## Installation

Run the commands from the relevant ET or MR model directory.


python3 -m pip install -r requirements.txt

The project uses TensorFlow/Keras for model training, FedML for federated learning, and Gurobi for solving the resource-allocation problem. Gurobi must be installed and licensed correctly.

## Models and datasets

All datasets should be placed in the `data/` directory in the root of the relevant model folder. Keep the training and test data in the locations expected by the data-loading scripts.

### VGG16

VGG16 is the image-classification model used in the RHiDE evaluation. It uses the CIFAR-10 image dataset, which contains 60,000 images.

- **Download:** CIFAR-10 is available from the official website: https://www.cs.toronto.edu/~kriz/cifar.html
- **Training:** Use the CIFAR-10 training data.
- **Inference/testing:** Use the CIFAR-10 test data.

Place the dataset under the model's `data/` directory, then prepare it using:


python scripts/prepare_cifar10_vgg16.py
python3 scripts/prepare_vgg16_val_npz.py

The second command converts the prepared validation data into the compact NPZ file used by the scripts.

### XEventNet

XEventNet is used to predict extreme weather events from Weather Research and Forecasting (WRF) simulation output.

- **Training:** Use labelled WRF output data, where the weather events are labelled for training.
- **Inference/testing:** Use WRF output files as input to the model.

Place the WRF output files and their labels under the model's `data/` directory. Make sure the paths and data format match the settings expected by the XEventNet data-loading scripts.

The VGG16 preparation commands above are only for CIFAR-10; they are not used to prepare XEventNet's WRF data.

## Configuration directories

The `configs_vgg16/` and corresponding XEventNet configuration directories contain YAML files for each experiment. Use the configuration files from the current ET or MR folder.

- `train.yaml`: model training and dataset settings.
- `federated.yaml`: device and edge-server setup, including their associations.
- `energy_latency.yaml`: device resource limits and per-split computation, communication, memory, and timing values.
- `split.yaml`: model split settings.
- `solver.yaml`: optimization settings for the Gurobi-based solver.
- `rl.yaml`: reinforcement-learning settings for adaptive aggregation.

Check dataset paths and resource values before running an experiment.

## Generating scenario YAML files

The scenario-generation scripts create scenario-specific YAML files from a scenario CSV. The CSV describes the scenario name, number of devices and edges, resource ranges, and whether the scenario is heterogeneous.

For XEventNet, the generation script is:


python3 scripts/generate_yamls_xeventnet_paper.py --scenarios scenarios_xeventnet.csv --outdir outxev


Use the VGG16 scenario-generation script and its matching CSV for VGG16. Adjust paths if those files are stored in another directory in your checkout.

Homogeneous scenarios use the same resource realization across devices. Heterogeneous scenarios assign different resource values to devices and/or edge servers. After generation, point the experiment configuration to the YAML files for the scenario you want to run.


## Scenario configuration files

The `outvgg/` and `outxevent/` directories contain the generated configurations for the scenarios in the paper. Each scenario has its own folder (for example, `S1/`, `S2/`, ..., `S12/`) containing two files:

- `energy_latency.yaml`: device-wise parameters, including resource limits and per-split computation, communication, memory, and timing values.
- `federated.yaml`: edge-wise parameters, including edge memory and energy budgets, device-to-edge assignments, and edge priorities. It also records the edge associated with each device.

The scenario-generation scripts create these files from the matching scenario CSV. The CSV specifies each scenario and its resource settings. Use the matching generator and CSV for VGG16 or XEventNet. Homogeneous scenarios use consistent resource settings across devices; heterogeneous scenarios introduce controlled differences between devices and/or edges.

## Running all scenarios

To run all generated VGG16 scenarios in one go, use:


python run_all_scenarios_vgg16.py --scenarios-dir outvgg --configs-dir configs_vgg16 --rounds 20 --extra "--seed 1"

This runner processes the scenario folders under `outvgg` using the configurations in `configs_vgg16`. For XEventNet, use its corresponding all-scenarios runner and configuration directory if available in the repository, and check its command-line arguments before running it.


