#!/bin/bash
# Build script for Dynamic Pick & Reorient workspace

set -e

echo "=== Dynamic Pick & Reorient Build Script ==="

# Check ROS2 installation
if [ -z "$ROS_DISTRO" ]; then
    echo "Error: ROS2 is not sourced. Please run: source /opt/ros/<distro>/setup.bash"
    exit 1
fi

echo "Using ROS2 distro: $ROS_DISTRO"

# Navigate to workspace root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
WORKSPACE_ROOT="$SCRIPT_DIR"

# If we're in src, go up one level
if [[ "$SCRIPT_DIR" == */src ]]; then
    WORKSPACE_ROOT="$(dirname "$SCRIPT_DIR")"
fi

cd "$WORKSPACE_ROOT"

echo "Workspace root: $WORKSPACE_ROOT"

# Install dependencies
echo ""
echo "=== Installing dependencies ==="
rosdep update
rosdep install --from-paths src --ignore-src -r -y

# Build the workspace
echo ""
echo "=== Building workspace ==="
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release

# Source the workspace
echo ""
echo "=== Build complete ==="
echo ""
echo "To use the workspace, run:"
echo "  source $WORKSPACE_ROOT/install/setup.bash"
echo ""
echo "To launch the demo:"
echo "  ros2 launch dynamic_reorient demo.launch.py"
echo ""
echo "Or launch components separately:"
echo "  Terminal 1: ros2 launch dynamic_reorient_gazebo gazebo.launch.py"
echo "  Terminal 2: ros2 launch dynamic_reorient_moveit move_group.launch.py"
echo "  Terminal 3: ros2 run dynamic_reorient pose_estimator"
echo "  Terminal 4: ros2 run dynamic_reorient pick_reorient_node"