# Dynamic Pick & Reorient

A ROS2 robotics project for autonomous pick-and-place of heterogeneous objects (bottles, boxes, cylinders) with vision-based 6D pose estimation and dynamic reorientation of horizontal objects into vertical placement.

## Project Overview

This project implements a robot capable of:
1. **Detecting objects** on a table using an overhead RGB-D camera
2. **Estimating 6D pose** via color segmentation, shape classification (bottle/box/cylinder), and orientation detection (vertical vs horizontal)
3. **Planning grasps** with per-object grip width and approach yaw
4. **Reorienting horizontal objects** mid-air through wrist joint rotation
5. **Placing objects** into color-coded containers with precise descent

### Hardware (Simulated)
- **Robot:** Universal Robots UR5 (6-DOF manipulator)
- **Gripper:** Robotiq 2F-85 parallel gripper
- **Sensor:** Overhead RGB-D camera (640x480, 15 Hz)

### Software Stack
- ROS2 Humble
- Gazebo Classic
- MoveIt2 (IK solver via `compute_ik` service)
- OpenCV + SciPy for vision processing
- `gazebo_ros2_control` for joint trajectory controllers
- Custom Gazebo grasp plugin (`grasp_attach` / `grasp_detach` services)

## Installation

### Prerequisites
```bash
# ROS2 Humble on Ubuntu 22.04
sudo apt install ros-humble-desktop

# MoveIt2
sudo apt install ros-humble-moveit

# Gazebo + ros2_control
sudo apt install ros-humble-gazebo-ros-pkgs ros-humble-gazebo-ros2-control

# Additional dependencies
sudo apt install ros-humble-controller-manager \
                 ros-humble-joint-state-broadcaster \
                 ros-humble-joint-trajectory-controller \
                 python3-opencv python3-scipy
```

### Build
```bash
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
This launches Gazebo, MoveIt, the pose estimator, and the pick & reorient node with appropriate delays.

### Step by Step
```bash
# Terminal 1: Gazebo simulation + controllers
ros2 launch dynamic_reorient gazebo.launch.py

# Terminal 2: MoveIt2 (move_group + RViz)
ros2 launch dynamic_reorient moveit.launch.py

# Terminal 3: Pose estimator (vision node)
ros2 run dynamic_reorient pose_estimator

# Terminal 4: Pick & reorient node
ros2 run dynamic_reorient pick_reorient_node
```

## Architecture

```
┌─────────────────┐     ┌──────────────────┐
│   RGB-D Camera  │────▶│  Pose Estimator  │
│  /camera/*      │     │  (color + shape) │
└─────────────────┘     └────────┬─────────┘
                                 │ /detected_objects (PoseStamped)
                                 │ /detected_markers (MarkerArray)
                                 ▼
┌─────────────────┐     ┌──────────────────┐     ┌──────────────┐
│     Gazebo      │◀───▶│ Pick & Reorient  │────▶│   MoveIt2    │
│   Simulation    │     │ (vision-driven)  │     │  /compute_ik │
└────────┬────────┘     └──────────────────┘     └──────────────┘
         │                   │         │
         │  grasp_attach/    │         │ /arm_controller
         │  grasp_detach     │         │ /gripper_controller
         └───────────────────┘         └──────────────────────▶ ros2_control
```

## Key Features

### 6D Pose Estimation (`pose_estimator`)
- HSV color-based segmentation for red (bottles), green (boxes), blue (cylinders)
- Shape classification via contour analysis (aspect ratio, solidity, circularity)
- Vertical vs horizontal orientation detection from top-down camera view
- Depth-based 3D back-projection with camera-to-world TF transformation
- Container exclusion filtering to avoid false detections on drop-off bins
- Debug image stream on `/pose_estimator/debug` for visualization

### Vision-Driven Pick & Place Pipeline (`pick_reorient_node`)
- **Closed-loop vision→pick**: subscribes to `/detected_objects` from the pose estimator
- Accumulates detections over a configurable time window (default 8s), then clusters by spatial proximity to deduplicate
- Automatically builds a task list with per-shape grasp parameters (grip width, approach yaw) from vision data
- Assigns container placement slots per color dynamically
- **Deterministic fallback**: if the pose estimator is unavailable or detects no objects, falls back to known object positions from the world file
- Processes vertical objects first, then horizontal (reorient) — sorted automatically from vision
- Incremental Cartesian descent via IK-solved waypoints (`_move_z`)
- IK seed continuity for smooth joint-space trajectories
- Grasp attach/detach via Gazebo plugin services for reliable object holding

### Dynamic Reorientation
- Horizontal-to-vertical reorientation via wrist_1_joint rotation (-90°)
- Orientation-preserving Cartesian moves after reorient (`_move_to_pose_tilted`)
- Tilted placement with configurable backoff for container insertion

### Container Placement
- Color-coded containers (red, green, blue) with per-container wall height offsets
- Configurable insert height accounting for container walls and object type
- Smooth multi-step descent into containers

## Configuration

### Object Detection
Edit `pose_estimator.py` to adjust:
- HSV color ranges per object class
- Minimum/maximum contour area thresholds
- Container exclusion positions and radius
- Detection rate (default: 2 Hz)

### Motion Planning
Edit `config/ompl_planning.yaml` for:
- Planner selection (RRTConnect, RRT*, PRM)
- Planning time limits
- Arm group projection evaluator

Edit `config/kinematics.yaml` for:
- IK solver plugin (default: LMAKinematicsPlugin)
- Solver timeout and attempts

### Controller Tuning
Edit `config/ur5_controllers.yaml` for:
- Controller update rate (default: 100 Hz)
- Joint trajectory goal tolerances
- State publish rates

## Troubleshooting

### Gazebo not starting
```bash
killall gzserver gzclient
ros2 launch dynamic_reorient gazebo.launch.py
```

### Controllers not activating
The controllers are activated 15 seconds after launch. If Gazebo is slow to load:
```bash
# Check if controller_manager is running
ros2 service list | grep controller_manager

# Manually activate controllers
ros2 control load_controller --set-state active joint_state_broadcaster
ros2 control load_controller --set-state active arm_controller
ros2 control load_controller --set-state active gripper_controller

# Verify
ros2 control list_controllers
```

### MoveIt IK failures
- Check joint limits in `config/joint_limits.yaml`
- Verify SRDF collision pairs in `config/ur5_gripper.srdf`
- The node uses `avoid_collisions = False` for IK — collisions are managed by task sequencing

### Camera not publishing
- Verify topics: `/camera/image_raw`, `/camera/depth/image_raw`, `/camera/camera_info`
- Check Gazebo camera plugin in the URDF xacro

## References

- [MoveIt2 Documentation](https://moveit.ros.org/)
- [Gazebo Tutorials](https://gazebosim.org/docs)
- [ROS2 Control](https://control.ros.org/)
- [Universal Robots ROS2 Description](https://github.com/UniversalRobots/Universal_Robots_ROS2_Description)
- Original inspiration: [TrashThrower Project](https://github.com/davidedavo/smart_robotics)

## License

MIT License

## Authors