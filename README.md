# Robot Dream : a framework for RL based autonomous navigation

This repository allows to train agents for autonomous navigation tasks. The agents proposed here are based on DREAMER a strong Model Based Reinforcement Learning (MBRL).
To simulate the robots and their environment we use NVIDIA's Isaac Sim, a GPU accelerated simulation software.  
On top of providing a DREAMER wrapper for Isaac, we also provide a Buoyancy plugin that allows to simulate hydrodynamic effects.
This plugin is a port of the UUV simulator, a well establish plugin in gazebo community to simulate the behaviors of Unmanned Surface Vehicles (USVs) as well as Autonomous Underwater Vehicles (AUVs).

## What's in the package?

The code provided here allows to train agents to solve various autonomous navigation tasks using a MBRL agent. Compared to previous release of our code, this repository does not use ROS for training and does not requires ROS to run the simulation at all. However, we provide the necessary tools to deploy the models inside ROS nodes!

We provide the following agents:
- Vanilla DREAMER (LINK).
- DREAMER with physical state (LINK).
- Goal Conditioned DREAMER with physical state.
- Goal Conditioned DREAMER with physical state and in imagination domain randomization.

We provide the following tasks:
- Shore following : following of a shore, or a side wall, at a given distance and velocity.
- Velocity conditioned shore following : following of a shore, or side wall, at a given distance. The velocity is given as a condition to the agent.
- Goal reaching : Tasking a system with reaching a state defined goal.
- Station keeping : Tasking a system to keep a steady position.
- Heading following with obstacle avoidance.

We provide the following environments:
- 9 procedurally generated lakes, both with solid ground and water. (LINK)
- 300 Environments similar to the BARN challenge, both with solid ground and water. (LINK)
- On the fly generation of BARN style environments, both with solid ground and water.

We provide the following robots:
- A ClearPath Kingfisher/Heron (LINK).
- A ClearPath Husky (LINK).
- A ClearPath Jackal (LINK).

This repository allows to reproduce the results presented in the following papers:
- ICRA: LINK
- CORL: LINK
- TRO: LINK


## Requirement

To train agents you will need an Nvidia GPU with Ray-Tracing. We recommended using GPUs with 12Gb of RAM or more (RTX 2080Ti, RTX3090, A5000, A6000).


You will also need to install Isaac Sim. To install Isaac Sim you can follow NVIDIA's tutorial here: LINK. We would recomment using their launcher as it makes the whole installation process easier that going with a docker. However, if you want to install it using the launcher, you will need a machine running ubuntu 18.04 (as of June 2022).

You will also need to download our RLEnvironments
