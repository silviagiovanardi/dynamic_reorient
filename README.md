# Dynamic Pick & Reorient

A ROS2 robotics project for autonomous manipulation of cylindrical and irregular objects with 6D pose estimation and dynamic reorientation.

## Project Overview

This project implements a robot capable of:
1. **Detecting objects** on a table using RGB-D camera
2. **Estimating 6D pose** of cylindrical/irregular objects
3. **Planning grasps** considering object orientation
4. **Reorienting objects** mid-air through waypoint-based trajectory
5. **Inserting objects** into precise slots while maintaining orientation constraints

### Hardware (Simulated)
- **Robot:** ABB IRB4600 (6-DOF industrial manipulator)
- **Gripper:** Robotiq 2F-85 parallel gripper
- **Sensor:** RGB-D camera (simulated depth camera)

### Software Stack
- ROS2 (Humble/Iron)
- Gazebo (Classic or Ignition)
- MoveIt2
- OpenCV for vision processing

## Installation

### Prerequisites
```bash
# ROS2 (example for Humble on Ubuntu 22.04)
sudo apt install ros-humble-desktop

# MoveIt2
sudo apt install ros-humble-moveit

# Gazebo
sudo apt install ros-humble-gazebo-ros-pkgs ros-humble-gazebo-ros2-control

# Additional dependencies
sudo apt install ros-humble-controller-manager \
                 ros-humble-joint-state-broadcaster \
                 ros-humble-joint-trajectory-controller \
                 python3-opencv python3-scipy
```

### Build
```bash
# Create workspace
mkdir -p ~/dynamic_reorient_ws/src
cd ~/dynamic_reorient_ws/src

# Clone or copy the packages here
# Then build
cd ~/dynamic_reorient_ws
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash
```

## Usage

### Full Demo
```bash
ros2 launch dynamic_reorient demo.launch.py
```

### Step by Step
```bash
# Terminal 1: Launch Gazebo simulation
ros2 launch dynamic_reorient_gazebo gazebo.launch.py

# Terminal 2: Launch MoveIt
ros2 launch dynamic_reorient_moveit move_group.launch.py

# Terminal 3: Launch pose estimator
ros2 run dynamic_reorient pose_estimator

# Terminal 4: Launch pick and reorient node
ros2 run dynamic_reorient pick_reorient_node
```

## Architecture

```
┌─────────────────┐     ┌──────────────────┐
│   RGB-D Camera  │────▶│  Pose Estimator  │
└─────────────────┘     └────────┬─────────┘
                                 │ /detected_object_pose
                                 ▼
┌─────────────────┐     ┌──────────────────┐
│     Gazebo      │◀───▶│ Pick & Reorient  │
│   Simulation    │     │      Node        │
└─────────────────┘     └────────┬─────────┘
                                 │
                                 ▼
                        ┌──────────────────┐
                        │     MoveIt2      │
                        │  Motion Planning │
                        └──────────────────┘
```

## Key Features

### 6D Pose Estimation
- Color-based object segmentation
- Ellipse fitting for orientation estimation
- Depth-based 3D position reconstruction
- Camera-to-base frame transformation

### Dynamic Reorientation
- SLERP-based orientation interpolation
- Waypoint generation for smooth rotation
- Orientation constraints during motion
- Path planning with collision avoidance

### Slot Insertion
- Precise positioning for narrow slots
- Vertical orientation maintenance
- Cartesian path planning for insertion

## Configuration

### Object Detection
Edit `pose_estimator.py` to adjust:
- Color ranges (HSV thresholds)
- Minimum/maximum object area
- Detection rate

### Motion Planning
Edit `config/ompl_planning.yaml` for:
- Planner selection (RRTConnect, RRT*, etc.)
- Planning time limits
- Path constraints

### Controller Tuning
Edit `config/controllers.yaml` for:
- PID gains
- Velocity/acceleration limits
- Trajectory tolerances

## Troubleshooting

### Gazebo not starting
```bash
# Kill any existing Gazebo processes
killall gzserver gzclient
# Try again
ros2 launch dynamic_reorient_gazebo gazebo.launch.py
```

### MoveIt planning failures
- Check joint limits in `joint_limits.yaml`
- Verify SRDF collision pairs
- Increase planning time in OMPL config

### Camera not publishing
- Check topic names in launch files
- Verify Gazebo camera plugin configuration

## Future Improvements

1. **Machine Learning Integration**
   - Train pose estimation network (e.g., PoseCNN, DenseFusion)
   - Reinforcement learning for grasp optimization

2. **Advanced Manipulation**
   - Force/torque feedback for grasp verification
   - Slip detection and recovery
   - Multi-object manipulation

3. **Real Robot Deployment**
   - Hardware interface for ABB controller
   - Real camera calibration
   - Safety monitoring

## References

- [MoveIt2 Documentation](https://moveit.ros.org/)
- [Gazebo Tutorials](https://gazebosim.org/docs)
- [ROS2 Control](https://control.ros.org/)
- Original inspiration: [TrashThrower Project](https://github.com/davidedavo/smart_robotics)

## License

MIT License

## Authors
