import os
import tempfile
import xacro
import re
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction, SetEnvironmentVariable
from launch_ros.actions import Node


def generate_launch_description():
    pkg = get_package_share_directory('dynamic_reorient')
    ur_desc = get_package_share_directory('ur_description')

    controllers_yaml = os.path.join(pkg, 'config', 'ur5_controllers.yaml')

    xacro_file = os.path.join(pkg, 'urdf', 'ur5_with_gripper.urdf.xacro')
    world_file = os.path.join(pkg, 'worlds', 'pick_reorient.world')
    fastrtps_profile = os.path.join(pkg, 'config', 'fastrtps_no_shm.xml')

    robot_description_content = xacro.process_file(
        xacro_file,
        mappings={'controllers_yaml': controllers_yaml},
    ).toxml()

    robot_description_content = re.sub(
        r'<ros2_control\b[^>]*>.*?<plugin>\s*ur_robot_driver/URPositionHardwareInterface\s*</plugin>.*?</ros2_control>',
        '',
        robot_description_content,
        flags=re.DOTALL
    )

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[
            {'robot_description': robot_description_content},
            {'use_sim_time': True},
        ],
    )
    
    static_world_to_base = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=['0', '0', '0', '0', '0', '0', 'world', 'base_link'],
        output='screen',
        parameters=[{'use_sim_time': True}],
    )

    urdf_tmp = tempfile.NamedTemporaryFile(
        mode='w', suffix='.urdf', prefix='ur5_gripper_', delete=False
    )
    urdf_tmp.write(robot_description_content)
    urdf_tmp.close()

    model_paths = [
        os.path.join(ur_desc, '..'),
        os.path.join(pkg, '..'),
    ]
    current = os.environ.get('GAZEBO_MODEL_PATH', '')
    full_path = ':'.join(model_paths) + (':' + current if current else '')

    set_model_path = SetEnvironmentVariable(
        name='GAZEBO_MODEL_PATH',
        value=full_path,
    )

    ws_root = os.path.normpath(os.path.join(os.path.dirname(pkg), '..', '..', '..'))
    plugin_paths = [
        os.path.join(ws_root, 'install', 'gazebo_grasp_fix', 'lib'),
        os.path.join(ws_root, 'install', 'gazebo_ros2_control', 'lib'),
    ]
    current_plugin = os.environ.get('GAZEBO_PLUGIN_PATH', '')
    full_plugin_path = ':'.join(plugin_paths) + (':' + current_plugin if current_plugin else '')

    set_plugin_path = SetEnvironmentVariable(
        name='GAZEBO_PLUGIN_PATH',
        value=full_plugin_path,
    )

    gzserver = ExecuteProcess(
        cmd=[
            'gzserver', '--verbose', world_file,
            '-s', 'libgazebo_ros_init.so',
            '-s', 'libgazebo_ros_factory.so',
        ],
        additional_env={
            'FASTRTPS_DEFAULT_PROFILES_FILE': fastrtps_profile,
            'GAZEBO_IP': '127.0.0.1',
            'GAZEBO_PLUGIN_PATH': full_plugin_path,
        },
        output='screen',
    )

    gzclient = ExecuteProcess(
        cmd=['gzclient'],
        output='screen',
    )


    # Spawn from the temp URDF file
    spawn = TimerAction(
        period=4.0,
        actions=[
            Node(
                package='gazebo_ros',
                executable='spawn_entity.py',
                arguments=[
                    '-entity', 'ur5',
                    '-file', urdf_tmp.name,
                    '-x', '0', '-y', '0', '-z', '0',
                ],
                output='screen',
            ),
        ],
    )

    # Activate controllers: wait for controller_manager, then load+activate
    activate_script = (
        'echo "Waiting for controller_manager..."; '
        'until ros2 service list 2>/dev/null | grep -q /controller_manager/list_controllers; do sleep 1; done; '
        'echo "controller_manager ready, activating controllers..."; '
        'for c in joint_state_broadcaster arm_controller gripper_controller; do '
        '  ros2 control load_controller --set-state active "$c" 2>/dev/null || '
        '  ros2 control set_controller_state "$c" active 2>/dev/null; '
        'done; '
        'echo "Controllers activated"; '
        'ros2 control list_controllers'
    )
    load_controllers = TimerAction(
        period=15.0,
        actions=[
            ExecuteProcess(
                cmd=['bash', '-c', activate_script],
                output='screen',
            ),
        ],
    )

    return LaunchDescription([
        set_model_path,
        set_plugin_path,
        robot_state_publisher,
        static_world_to_base,
        gzserver,
        gzclient,
        spawn,
        load_controllers,
    ])
