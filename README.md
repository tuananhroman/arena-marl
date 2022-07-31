# ARENA MARL
Arena marl is a package from the ***Arena ROSNav*** organisation. Here we develop all modular functionalities for a **M**ulti **A**gent **R**einforcement **L**earning setting. This includes homogeneous and heterogeneous setups!

## Installation and setup

First you have to create a standard catkin workspace

```bash
mkdir MARL_ws && mkdir MARL_ws/src
```

Now clone this repository into the src directory
```bash
cd MARL_ws/src
git clone git@github.com:tuananhroman/arena-marl.git
```

Install and activate the poetry environment
```bash
cd arena-marl
poetry install && poetry shell
```

Add all additional necessary packages via the .rosinstall
```bash
rosws update
```

Build the whole catkin workspace
```bash
cd ../..
catkin_make -DCMAKE_BUILD_TYPE=Release -DPYTHON_EXECUTABLE=/usr/bin/python3 -DCMAKE_CXX_STANDARD=14
```

Source the correct setup path.
> When using bash use .bash/ .bashrc and when using zsh use .zsh/ .zshrc
```bash
source $HOME/MARL_ws/devel/setup.zsh
```

You are good to go!

## Example Usage

### Deployment

1. In the first terminal launch all necessary nodes.
    > num_robots is the number off all agents combined 
    ```bash
    roslaunch arena_bringup start_evaluation.launch map_folder_name:=map_small num_robots:=4     
    ```

2. Define deployment setup in [deployment_config.yaml](training/configs/deployment_config.yaml).
    > You can here ad as many robots with their respective agents as you like!

3. In the second terminal start the deployment script.
    ```bash
    python marl_deployment.py --config deployment_config.yaml     
    ```

4. *(OPTIONAL)* In a third terminal start the visualization.
    ```bash
    roslaunch arena_bringup visualization_training.launch rviz_file:=nav_MARL
    ```

## Trouble shooting
If you have some issues with packages not being found try exporting the following paths:
```bash
export ROS_PACKAGE_PATH=$HOME/MARL_ws/src:/opt/ros/noetic/share
export PYTHONPATH=$HOME/MARL_ws/src/arena-marl:$HOME/MARL_ws/devel/lib/python3/dist-packages:/opt/ros/noetic/lib/python3/dist-packages:${PYTHONPATH}
```
