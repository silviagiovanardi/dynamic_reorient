#!/usr/bin/env python3
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    pkg = get_package_share_directory('dynamic_reorient')

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg, 'launch', 'gazebo.launch.py'))
    )

    moveit = TimerAction(
        period=10.0,
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(pkg, 'launch', 'moveit.launch.py'))
            ),
        ],
    )

    pose_estimator = TimerAction(
        period=14.0,
        actions=[
            Node(
                package='dynamic_reorient',
                executable='pose_estimator',
                output='screen',
                parameters=[{'use_sim_time': True}],
            ),
        ],
    )

    pick_node = TimerAction(
        period=30.0,
        actions=[
            Node(
                package='dynamic_reorient',
                executable='pick_reorient_node',
                output='screen',
                parameters=[{'use_sim_time': True}],
            ),
        ],
    )

    # Camera debug view — shows what the pose estimator sees
    camera_view = TimerAction(
        period=16.0,
        actions=[
            ExecuteProcess(
                cmd=['ros2', 'run', 'rqt_image_view', 'rqt_image_view', '/pose_estimator/debug'],
                output='screen',
            ),
        ],
    )

    return LaunchDescription([gazebo, moveit, pose_estimator, pick_node, camera_view])
