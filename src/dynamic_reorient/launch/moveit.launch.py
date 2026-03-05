#!/usr/bin/env python3
import os
import re
import yaml
import xacro
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def load_yaml(pkg, path):
    full_path = os.path.join(get_package_share_directory(pkg), path)
    with open(full_path, 'r') as f:
        return yaml.safe_load(f)


def generate_launch_description():
    pkg = get_package_share_directory('dynamic_reorient')

    xacro_file = os.path.join(pkg, 'urdf', 'ur5_with_gripper.urdf.xacro')
    robot_description_content = xacro.process_file(xacro_file).toxml()

    robot_description_content = re.sub(
        r'<ros2_control\b[^>]*>.*?<plugin>\s*ur_robot_driver/URPositionHardwareInterface\s*</plugin>.*?</ros2_control>',
        '',
        robot_description_content,
        flags=re.DOTALL
    )
    robot_description = {'robot_description': robot_description_content}

    srdf_file = os.path.join(pkg, 'srdf', 'ur5_gripper.srdf')
    with open(srdf_file, 'r') as f:
        robot_description_semantic = {'robot_description_semantic': f.read()}

    kinematics = load_yaml('dynamic_reorient', 'config/kinematics.yaml')

    joint_limits = {
        'robot_description_planning': load_yaml('dynamic_reorient', 'config/joint_limits.yaml')
    }

    ompl = load_yaml('dynamic_reorient', 'config/ompl_planning.yaml')
    controllers = load_yaml('dynamic_reorient', 'config/moveit_controllers.yaml')
    ompl_config = {'move_group': ompl}

    move_group = Node(
        package='moveit_ros_move_group',
        executable='move_group',
        output='screen',
        parameters=[
            robot_description,
            robot_description_semantic,
            kinematics,
            joint_limits,
            ompl_config,
            controllers,
            {'use_sim_time': True},
            {'publish_robot_description_semantic': True},
        ],
    )

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        output='screen',
        parameters=[
            robot_description,
            robot_description_semantic,
            kinematics,
            {'use_sim_time': True},
        ],
    )

    return LaunchDescription([move_group, rviz])
