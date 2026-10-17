# RHiDE: Resource-adaptive Hierarchical Distributed Learning on the Edge

## Overview

RHiDE is a resource-aware framework for hierarchical split learning in IoT, edge, and cloud environments.

IoT devices have limited energy, memory, and computing power. Because of these limitations, it may not always be possible for a device to perform all the required training. RHiDE addresses this problem by deciding how much of a deep learning model should run on the device and how much should run on the edge server.

RHiDE also decides which devices should participate in training and when raw data should be sent to the edge server if local training is not possible. These decisions consider device energy, device memory, edge energy, edge memory, communication cost, and execution time.

RHiDE uses a Mixed-Integer Linear Programming (MILP) formulation and has two versions: **RHiDE-ET** and **RHiDE-MR**.

## RHiDE-ET

RHiDE-ET makes optimization decisions for one training round at a time. It selects the model split point and decides which devices can participate based on the available resources.
The main aim is to reduce resource consumption while meeting the execution-time and other resource constraints.

## RHiDE-MR

RHiDE-MR considers multiple training rounds instead of optimizing only the current round. It considers the remaining energy of devices and tries to preserve energy so that devices can continue participating in future rounds. The model uses a planned training horizon to make these decisions. If the available resources are not sufficient to complete all the planned rounds, the number of completed rounds may be lower than the requested horizon.

## Deep Learning Models

RHiDE is evaluated using two models: VGG16 and XEventNet.

### VGG16

VGG16 is used for image classification with the CIFAR-10 dataset. It has large intermediate activations and relatively high computation costs. It is used to study how RHiDE selects model split points under different device and edge resource constraints.

### XEventNet

XEventNet is used for extreme weather event prediction using Weather Research and Forecasting (WRF) simulation output. Compared with VGG16, XEventNet has lower activation sizes and computational requirements.

For training, labelled WRF output data can be used. WRF output files can also be used for inference. Dataset preparation and execution details are available in the README inside each model directory.

## Experimental Scenarios

The paper uses 12 scenarios to evaluate RHiDE under different resource settings.

- **S1–S6: Homogeneous scenarios.** Devices use the same resource settings within a scenario.
- **S7–S12: Heterogeneous scenarios.** Devices may have different energy capacities, memory limits, communication bandwidths, and computational capabilities.

The scenarios also vary the energy and memory available at the edge servers. Most scenarios use 100 IoT devices, while S6 and S12 use 500 devices. The experiments use 10 edge servers.

## Repository Structure

The repository contains separate directories for the two RHiDE models. Each model directory contains the source code, configuration files, and scripts for that model.


## Installation and Running the Models

The installation steps, dependencies, dataset preparation commands, and experiment commands are provided in the README files inside the two model directories.

- [RHiDE-ET README](RHIDE_ET/README.md)
- [RHiDE-MR README](RHIDE_MR/README.md)

