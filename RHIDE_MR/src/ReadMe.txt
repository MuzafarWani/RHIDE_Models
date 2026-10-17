accounting/
Handles runtime measurement of resource usage during training and inference.
It records how much energy, latency, and upload each device actually consumes in a round.
Components:
energy_latency_log.py – Accumulates per-device energy and latency and publishes telemetry.
upload_counter.py – Tracks bytes uploaded by each device.
Details.txt – Human-readable notes for this module.

data/
Contains all data preparation and preprocessing utilities.
It loads raw datasets, builds grids/windows, normalizes values, and converts data into training-ready tensors.
Components:
dataset_builder.py – High-level dataset construction pipeline.
encoders.py – Label encoders and decoding utilities.
grid_builder.py – Extracts spatial windows from WRF or grid data.
normalization.py – Feature normalization helpers.
tf_dataset.py – TensorFlow dataset builders.
wrf_reader.py – Reads WRF weather data files.
Details.txt – Module description.

eval/
Provides evaluation utilities for trained models.
It computes metrics and generates summary reports for experiments.
Components:
metrics.py – Evaluation metrics.
reports.py – Report and summary generation.

federated/
Implements the full hierarchical federated system: devices, edges, cloud, and communication.
This directory simulates how data, activations, and model updates move across layers.
Components:
accounting/ – Runtime energy, latency, and upload tracking.
device_client/ – Device-side training and split execution.
edge_server/ – Edge-side aggregation and split completion.
cloud_server/ – Global aggregation and model distribution.
messaging/ – In-process pub/sub bus and message formats.
priority.py – Maps devices to edges and exposes edge priorities.
details.txt – Overview of the federated layer.

inference/
split and full-model inference.It builds inference tensors, runs prediction with or without splits, and exports results.
Components:
build_grids.py – Builds 2D/3D input windows for inference.
predictor.py – Runs full or split inference using trained models.
postprocess.py – Attaches metadata and saves predictions to CSV.
details.txt – Module description.

models/
Defines all neural network models and model utilities.
It includes XEvenetNet, VGG16, split-graph builders, and serialization helpers.
Components:
xeventnet.py – Two-branch 2D+3D XEvenetNet model.
vgg16.py – VGG16-based classification model.
split_graph.py – Builds split device/edge submodels (XEvenetNet).
split_graph_vgg16.py – Split logic for VGG16.
metrics_losses.py – Custom metrics and loss functions.
serialization.py – Save/load and weight transfer utilities.
details.txt – Module description.

solver/
Implements the optimization core of SPHERE.
It builds round coefficients, selects split points, solves the MILP, and post-processes results.
Components:
beta_selection.py – Heuristic β selection by minimum energy.
gurobi_solver.py – Main MILP formulation and solver.
mappings.py – Canonical β definitions for XEvenetNet.
mappings_vgg16.py – β definitions for VGG16.
param_feeds.py – Builds numeric coefficients for each round.
postprocess.py – Summarizes solver output.
warm_start.py – Warm-start support for the solver.
Details.txt – Module description.

training/
Orchestrates the entire SPHERE training process.
It schedules batches, runs the solver, executes rounds, computes rewards, logs results, and checkpoints state.
Components:
train_loop.py – Main end-to-end training loop.
round_scheduler.py – Builds per-round device batches.
constraints.py – Post-solver sanity checks.
reward.py – Computes RL rewards.
logging_hooks.py – CSV and JSON logging.
checkpoints.py – Round-level checkpointing.
details.txt – Module description.

rl/
Implements reinforcement learning agents and environments.
Used to adapt aggregation cadence and other control parameters during training.
Components:
agents.py – PPO and DQN agents.
madrl_env.py – Lightweight environment wrapper.
replay.py – Replay buffer for off-policy RL.

utils/
General helper utilities shared across the project.
Components:
io_helpers.py – File and I/O helpers.
seeding.py – Random seed control.
timers.py – Timing utilities.
typing_aliases.py – Common type aliases.
