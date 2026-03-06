#!/usr/bin/env python3
"""
Pick and Reorient Node - DETERMINISTIC approach.

Uses KNOWN object positions from the world file.
Objects are processed in a fixed order: vertical first, horizontal last.
Uses grasp_attach to lock objects to gripper reliably.
"""
import time
import threading
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
import numpy as np
from scipy.spatial.transform import Rotation as R

from geometry_msgs.msg import PoseStamped, Point, Quaternion
from std_msgs.msg import String
from std_srvs.srv import SetBool
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.action import FollowJointTrajectory
from moveit_msgs.srv import GetPositionIK
from moveit_msgs.msg import MoveItErrorCodes
from builtin_interfaces.msg import Duration


# Each task: known position, known target, known grip
# Order: all vertical first, then horizontal
TASKS = [
    # === VERTICAL OBJECTS FIRST ===
    {
        'name': 'red_bottle_2 (standing)',
        'pick': {'x': 0.40, 'y': 0.30, 'z': 0.77},
        'yaw': 0.2,
        'shape': 'bottle', 'vertical': True,
        'grip': 0.45,  # bottle r=0.025, diameter=5cm
        'place': {'x': 0.55, 'y': 0.25},
        'color': 'red',
    },
    {
        'name': 'green_box_1 (standing)',
        'pick': {'x': 0.35, 'y': 0.10, 'z': 0.76},
        'yaw': 0.15,
        'shape': 'box', 'vertical': True,
        'grip': 0.42,  # box 5cm wide
        'place': {'x': 0.55, 'y': 0.0},
        'color': 'green',
    },
    {
        'name': 'green_box_2 (small)',
        'pick': {'x': 0.40, 'y': -0.10, 'z': 0.73},
        'yaw': 0.7,
        'shape': 'box', 'vertical': True,
        'grip': 0.50,  # small box 4cm wide
        'place': {'x': 0.65, 'y': 0.0},
        'color': 'green',
    },
    {
        'name': 'blue_cylinder_2 (standing)',
        'pick': {'x': 0.40, 'y': -0.35, 'z': 0.75},
        'yaw': 0.0,
        'shape': 'cylinder', 'vertical': True,
        'grip': 0.50,  # cylinder r=0.022, diameter=4.4cm
        'place': {'x': 0.55, 'y': -0.25},
        'color': 'blue',
    },
    # === HORIZONTAL OBJECTS ===
    {
        'name': 'red_bottle_1 (lying)',
        'pick': {'x': 0.28, 'y': 0.48, 'z': 0.735},
        'yaw': 0.4,
        'shape': 'bottle', 'vertical': False,
        'grip': 0.45,  # same as standing bottle
        'place': {'x': 0.65, 'y': 0.25},
        'color': 'red',
        'reorient': True,
    },
    {
        'name': 'blue_cylinder_1 (lying)',
        'pick': {'x': 0.25, 'y': -0.42, 'z': 0.73},
        'yaw': 0.8,
        'shape': 'cylinder', 'vertical': False,
        'grip': 0.50,  # same as standing cylinder
        'place': {'x': 0.65, 'y': -0.25},
        'color': 'blue',
        'reorient': True,
    },
]


class PickReorientNode(Node):
    TABLE_Z = 0.71
    CONTAINER_Z = 0.712
    SAFE_Z = 1.05
    REORIENT_Z = 1.15
    FINGER_ABOVE_TCP = 0.030

    CONTAINER_WALL = {'red': 0.08, 'green_big': 0.07, 'green_small': 0.05, 'blue': 0.06}

    def __init__(self):
        super().__init__('pick_reorient_node')
        self.cb_group = ReentrantCallbackGroup()

        self.arm_joints = [
            'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
            'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
        ]
        self.gripper_joints = [
            'finger_joint', 'right_outer_knuckle_joint',
            'left_inner_knuckle_joint', 'right_inner_knuckle_joint',
            'left_inner_finger_joint', 'right_inner_finger_joint',
        ]

        self.current_joints = None
        self.last_arm_seed = None
        self.task_index = 0
        self.is_busy = False
        self.controllers_ready = False

        self.arm_client = ActionClient(
            self, FollowJointTrajectory,
            '/arm_controller/follow_joint_trajectory',
            callback_group=self.cb_group)
        self.gripper_client = ActionClient(
            self, FollowJointTrajectory,
            '/gripper_controller/follow_joint_trajectory',
            callback_group=self.cb_group)
        self.ik_client = self.create_client(
            GetPositionIK, '/compute_ik', callback_group=self.cb_group)
        self.attach_srv = self.create_client(
            SetBool, '/grasp_attach', callback_group=self.cb_group)
        self.detach_srv = self.create_client(
            SetBool, '/grasp_detach', callback_group=self.cb_group)

        self.create_subscription(JointState, '/joint_states', self.joint_cb, 10)
        self.status_pub = self.create_publisher(String, '/pick_reorient/status', 10)

        self.create_timer(5.0, self.main_loop)
        self.get_logger().info('Pick & Reorient Node started (DETERMINISTIC)')

        self._home_timer = self.create_timer(
            5.0, self._go_home_once, callback_group=self.cb_group)

    def _wait_for_future(self, future, timeout_sec=10.0):
        event = threading.Event()
        future.add_done_callback(lambda _: event.set())
        return event.wait(timeout=timeout_sec)

    def joint_cb(self, msg):
        self.current_joints = msg

    def _go_home_once(self):
        self._home_timer.cancel()
        self.get_logger().info('Waiting for controllers...')
        self.arm_client.wait_for_server()
        self.gripper_client.wait_for_server()
        self.get_logger().info('Controllers ready')
        self.controllers_ready = True
        self.go_home()

    def go_home(self):
        self.publish_status('Moving to home')
        self.move_arm([0.0, -1.5708, 1.5708, -1.5708, -1.5708, 0.0], 3.0)

    def main_loop(self):
        if not self.controllers_ready or self.is_busy:
            return
        if self.task_index >= len(TASKS):
            self.get_logger().info('All tasks done!', throttle_duration_sec=30.0)
            return

        self.is_busy = True
        task = TASKS[self.task_index]
        try:
            self.execute_task(task)
            self.task_index += 1
        except Exception as e:
            self.get_logger().error(f'Task failed: {e}')
            import traceback
            traceback.print_exc()
            self.task_index += 1  # skip failed task
            time.sleep(2.0)
            self.go_home()
        self.is_busy = False

    # ------------------------------------------------------------------
    def make_pose(self, x, y, z, yaw=0.0):
        pose = PoseStamped()
        pose.header.frame_id = 'world'
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position = Point(x=x, y=y, z=z)
        pitch = np.pi - 0.05
        q = R.from_euler('xyz', [pitch, 0.0, yaw]).as_quat()
        pose.pose.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
        return pose

    def _move_z(self, x, y, z_start, z_end, yaw=0.0, steps=5, dur=0.8):
        for z in np.linspace(z_start, z_end, steps):
            if not self.move_to_pose(self.make_pose(x, y, float(z), yaw=yaw), duration=dur):
                return False
        return True

    def _insert_z(self, task):
        color = task['color']
        if color == 'green':
            key = 'green_big' if task['place']['x'] < 0.60 else 'green_small'
        else:
            key = color
        wall_h = self.CONTAINER_WALL.get(key, 0.08)
        return self.CONTAINER_Z + wall_h + 0.04

    # ------------------------------------------------------------------
    # MAIN TASK EXECUTION
    # ------------------------------------------------------------------
    def execute_task(self, task):
        p = task['pick']
        px, py, pz = p['x'], p['y'], p['z']
        yaw = task['yaw']
        grip = task['grip']
        place = task['place']
        color = task['color']
        needs_reorient = task.get('reorient', False)

        grasp_z = pz + 0.02
        grasp_z = max(grasp_z, self.TABLE_Z + 0.03)

        self.publish_status(f'=== {task["name"]} ===')

        # 1. Open gripper
        self.control_gripper(open_gripper=True)
        time.sleep(0.5)

        # 2. Move to safe height above object
        self.publish_status(f'Above object ({px:.2f}, {py:.2f})')
        if not self.move_to_pose(self.make_pose(px, py, self.SAFE_Z, yaw=yaw), duration=2.5):
            self.get_logger().error('Move to safe Z failed')
            return

        # 3. Descend above object
        approach_z = grasp_z + 0.10
        if not self.move_to_pose(self.make_pose(px, py, approach_z, yaw=yaw), duration=1.5):
            self.get_logger().error('Approach failed')
            return

        # 4. Final descent to grasp
        if not self._move_z(px, py, approach_z, grasp_z, yaw=yaw, steps=4, dur=0.6):
            self.get_logger().error('Descent failed')
            return

        # 5. GRASP: close partially → attach → close full
        time.sleep(0.3)
        pre_grip = max(grip - 0.10, 0.10)
        self.control_gripper_partial(pre_grip, duration=2)
        time.sleep(0.5)
        self.grasp_attach()
        time.sleep(0.5)
        self.control_gripper_partial(grip, duration=2)
        time.sleep(0.5)

        # 6. Lift
        self.publish_status('Lifting')
        if not self._move_z(px, py, grasp_z, self.SAFE_Z, yaw=yaw, steps=3, dur=1.0):
            self.get_logger().error('Lift failed')
            return

        # 7. Go home (safe config, object attached)
        self.go_home()
        time.sleep(0.5)

        # 8. Reorient if needed
        if needs_reorient:
            self.publish_status('Reorienting H->V')
            if not self.move_to_pose(self.make_pose(0.3, 0.0, self.REORIENT_Z, yaw=0.0), duration=2.0):
                self.get_logger().warn('Reorient lift failed, trying place anyway')
            else:
                self._reorient_wrist(tilt=-np.pi/2)
                time.sleep(0.5)
            self.go_home()
            time.sleep(0.5)

        # 9. Move above container
        self.publish_status(f'Above {color} container')
        if not self.move_to_pose(
            self.make_pose(place['x'], place['y'], self.SAFE_Z, yaw=0.0), duration=2.5):
            self.get_logger().error('Move above container failed')
            return

        # 10. Descend to release
        insert_z = self._insert_z(task)
        if not self._move_z(place['x'], place['y'], self.SAFE_Z, insert_z,
                            yaw=0.0, steps=3, dur=1.0):
            self.get_logger().error('Container descent failed')
            return

        # 11. Release
        self.grasp_detach()
        time.sleep(0.3)
        self.control_gripper(open_gripper=True)
        time.sleep(0.5)

        # 12. Retract
        self.move_to_pose(
            self.make_pose(place['x'], place['y'], self.SAFE_Z, yaw=0.0), duration=1.5)
        self.go_home()
        self.publish_status(f'Done: {task["name"]}')

    # ------------------------------------------------------------------
    def _reorient_wrist(self, tilt=-np.pi/2):
        if self.current_joints is None:
            return False
        jnames = list(self.current_joints.name)
        jpos = list(self.current_joints.position)

        def get_j(name):
            try:
                return jpos[jnames.index(name)]
            except ValueError:
                return 0.0

        joints = [
            get_j('shoulder_pan_joint'),
            get_j('shoulder_lift_joint'),
            get_j('elbow_joint'),
            get_j('wrist_1_joint') + tilt,
            get_j('wrist_2_joint'),
            get_j('wrist_3_joint'),
        ]
        return self.move_arm(joints, duration=3.0)

    # ------------------------------------------------------------------
    def grasp_attach(self):
        if not self.attach_srv.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn('grasp_attach not available')
            return
        req = SetBool.Request()
        req.data = True
        try:
            result = self.attach_srv.call(req)
            if result:
                self.get_logger().info(f'Attach: {result.message}')
        except Exception as e:
            self.get_logger().error(f'Attach failed: {e}')

    def grasp_detach(self):
        if not self.detach_srv.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn('grasp_detach not available')
            return
        req = SetBool.Request()
        req.data = True
        try:
            result = self.detach_srv.call(req)
            if result:
                self.get_logger().info(f'Detach: {result.message}')
        except Exception as e:
            self.get_logger().error(f'Detach failed: {e}')

    # ------------------------------------------------------------------
    def move_to_pose(self, target, duration=4.0):
        if not self.ik_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().error('IK not available')
            return False

        req = GetPositionIK.Request()
        req.ik_request.group_name = 'arm'
        req.ik_request.ik_link_name = 'tcp'
        req.ik_request.pose_stamped = target
        req.ik_request.timeout.sec = 5
        req.ik_request.avoid_collisions = False

        if self.last_arm_seed is not None:
            arm_js = JointState()
            arm_js.name = list(self.arm_joints)
            arm_js.position = list(self.last_arm_seed)
            req.ik_request.robot_state.joint_state = arm_js
        elif self.current_joints is not None:
            arm_js = JointState()
            arm_js.header = self.current_joints.header
            for i, name in enumerate(self.current_joints.name):
                if name in self.arm_joints:
                    arm_js.name.append(name)
                    arm_js.position.append(self.current_joints.position[i])
            req.ik_request.robot_state.joint_state = arm_js

        try:
            result = self.ik_client.call(req)
        except Exception as e:
            self.get_logger().error(f'IK exception: {e}')
            return False

        if result is None or result.error_code.val != MoveItErrorCodes.SUCCESS:
            p = target.pose.position
            code = result.error_code.val if result else 'None'
            self.get_logger().error(f'IK error {code} for ({p.x:.3f},{p.y:.3f},{p.z:.3f})')
            return False

        positions = []
        for jn in self.arm_joints:
            if jn in result.solution.joint_state.name:
                idx = result.solution.joint_state.name.index(jn)
                positions.append(result.solution.joint_state.position[idx])
        if len(positions) != 6:
            self.get_logger().error('Incomplete IK solution')
            return False

        return self.move_arm(positions, duration)

    def move_arm(self, positions, duration=4.0):
        if not self.arm_client.wait_for_server(timeout_sec=10.0):
            return False
        traj = JointTrajectory()
        traj.joint_names = self.arm_joints
        pt = JointTrajectoryPoint()
        pt.positions = positions
        pt.velocities = [0.0] * 6
        pt.time_from_start = Duration(
            sec=int(duration), nanosec=int((duration % 1) * 1e9))
        traj.points.append(pt)
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj
        future = self.arm_client.send_goal_async(goal)
        if not self._wait_for_future(future, timeout_sec=15.0):
            return False
        gh = future.result()
        if not gh or not gh.accepted:
            return False
        rf = gh.get_result_async()
        if not self._wait_for_future(rf, timeout_sec=duration + 20.0):
            try:
                gh.cancel_goal_async()
            except Exception:
                pass
            time.sleep(2.0)
            return False
        res = rf.result()
        if res is None or res.result.error_code != 0:
            return False
        self.last_arm_seed = list(positions)
        return True

    def control_gripper(self, open_gripper=True):
        if not self.gripper_client.wait_for_server(timeout_sec=5.0):
            return False
        traj = JointTrajectory()
        traj.joint_names = self.gripper_joints
        val = 0.0 if open_gripper else 0.8
        pt = JointTrajectoryPoint()
        pt.positions = [val, val, val, val, -val, -val]
        pt.velocities = [0.0] * 6
        pt.time_from_start = Duration(sec=2)
        traj.points.append(pt)
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj
        future = self.gripper_client.send_goal_async(goal)
        self._wait_for_future(future, timeout_sec=10.0)
        gh = future.result()
        if gh and gh.accepted:
            rf = gh.get_result_async()
            self._wait_for_future(rf, timeout_sec=15.0)
        return True

    def control_gripper_partial(self, value, duration=3):
        if not self.gripper_client.wait_for_server(timeout_sec=5.0):
            return False
        traj = JointTrajectory()
        traj.joint_names = self.gripper_joints
        pt = JointTrajectoryPoint()
        pt.positions = [value, value, value, value, -value, -value]
        pt.velocities = [0.0] * 6
        pt.time_from_start = Duration(sec=duration)
        traj.points.append(pt)
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj
        future = self.gripper_client.send_goal_async(goal)
        if not self._wait_for_future(future, timeout_sec=10.0):
            return False
        gh = future.result()
        if gh and gh.accepted:
            rf = gh.get_result_async()
            self._wait_for_future(rf, timeout_sec=duration + 10.0)
        return True

    def publish_status(self, msg):
        self.status_pub.publish(String(data=msg))
        self.get_logger().info(msg)


def main(args=None):
    rclpy.init(args=args)
    node = PickReorientNode()
    from rclpy.executors import MultiThreadedExecutor
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()