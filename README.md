# Thesis Inference Code

This repository contains experiment runner code from a master's thesis on inference workloads for sports video processing.

The code is organized around two fixed pretrained workloads:

* YOLO object detection on soccer frame data
* SportSBD shot boundary detection on sports broadcast video clips

The repository includes the main Python runner entry points, shared runtime utilities, workload-specific inference and evaluation helpers, and Docker environment files used for the experiment environments.

## Repository Structure

docker/
Dockerfile.yolo
Dockerfile.sportsbd
requirements-yolo.txt
requirements-sportsbd.txt

project/
src/
run\_yolo\_experiment.py
run\_sportsbd\_experiment.py
common/
sportsbd/
yolo/

## Main Runner Files

project/src/run\_yolo\_experiment.py
project/src/run\_sportsbd\_experiment.py

The YOLO runner loads a pretrained Ultralytics YOLO checkpoint, collects image inputs, runs prediction, saves structured prediction files, and computes object detection metrics.

The SportSBD runner loads the fixed SportSBD checkpoint bundle, reads video inputs, builds temporal clips, runs shot-boundary inference, saves predictions, and computes transition metrics.

## Shared Utilities

The shared common package contains utilities for:

* command-line configuration
* JSON and CSV output writing
* run directory creation
* runtime timing
* environment metadata
* host-level system monitoring

Each experiment run writes structured outputs with configuration snapshots, run metadata, runtime metrics, prediction files, and monitoring summaries where available.

## Docker Files

The Docker files describe the software environments used for the two workloads:

docker/Dockerfile.yolo
docker/Dockerfile.sportsbd

The corresponding dependency lists are:

docker/requirements-yolo.txt
docker/requirements-sportsbd.txt



## Model Sources



The experiment runners expect local checkpoint paths for the two pretrained workloads.



The YOLO object detection checkpoint used with the SoccerSum workload is available from the public SoccerSum Hugging Face repository:



https://huggingface.co/SimulaMet-HOST/SoccerSum/blob/main/detection.pt



The SportSBD checkpoint is provided through the public `sportsbd` package and repository:



https://pypi.org/project/sportsbd/



https://github.com/mehdih7/sportsbd



