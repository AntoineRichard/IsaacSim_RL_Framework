# Robot Dream : a framework for RL based autonomous navigation

This repository allows to train agents for autonomous navigation tasks. The agents proposed here are based on DREAMER a strong Model Based Reinforcement Learning (MBRL).
To simulate the robots and their environment we use NVIDIA's Isaac Sim, a GPU accelerated simulation software.  
On top of providing a DREAMER wrapper for Isaac, we also provide a Buoyancy plugin that allows to simulate hydrodynamic effects.
This plugin is a port of the UUV simulator, a well establish plugin in gazebo community to simulate the behaviors of Unmanned Surface Vehicles (USVs) as well as Autonomous Underwater Vehicles (AUVs).

## Requirement

To train agents you will need an Nvidia GPU with Ray-Tracing. We recommended using GPUs with 12Gb of RAM or more (RTX 2080Ti, RTX3090, A5000, A6000).
