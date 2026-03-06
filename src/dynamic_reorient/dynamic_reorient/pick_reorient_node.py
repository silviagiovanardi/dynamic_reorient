#!/usr/bin/env python3
"""
Pick and Reorient Node - Main manipulation controller.

Uses a Gazebo link-attacher plugin for reliable grasping:
  - /grasp_attach : creates a fixed joint between gripper and nearest object
  - /grasp_detach : removes the fixed joint

For horizontal objects (bottles/cylinders lying down):
  1. Pick while lying
  2. Lift high
  3. Rotate wrist so object becomes vertical
  4. Move to container and release

Motion strategy (collision-safe):
  - Always travel horizontally at SAFE_Z (0.35m above table)
  - Descend to target Z in small interpolated steps
  - Same pattern for both pick and place
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


class PickReorientNode(Node):
    def __init__(self):
        super().__init__('pick_reorient_node')

        self.cb_group = ReentrantCallbackGroup()

        self.arm_joints = [
            'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
            'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
        ]
        self.gripper_joints = [
            'finger_joint',
            'right_outer_knuckle_joint',
            'left_inner_knuckle_joint',
            'right_inner_knuckle_joint',
            'left_inner_finger_joint',
            'right_inner_finger_joint',
        ]

        # Container slots — must match world file positions
        self.color_slots = {
            'red':   [
                {'x': 0.55, 'y':  0.25, 'used': False},
                {'x': 0.65, 'y':  0.25, 'used': False},
            ],
            'green': [
                {'x': 0.55, 'y':  0.0,  'used': False},
                {'x': 0.65, 'y':  0.0,  'used': False},
            ],
            'blue':  [
                {'x': 0.55, 'y': -0.25, 'used': False},
                {'x': 0.65, 'y': -0.25, 'used': False},
            ],
        }

        self.current_joints = None
        self.detected_objects = []
        self.last_arm_seed = None
        self.is_busy = False
        self.controllers_ready = False

        # Gripper closing values (Robotiq 2F-85, 0.0=open, 0.8=full close)
        # Depends on shape AND orientation:
        #   V (vertical/standing) = top-down view (small cross-section)
        #   H (horizontal/lying)  = side grasp (full diameter/width)
        self.grip_values = {
            ('bottle', True):    {'partial': 0.55, 'full': 0.72},  # V: grabbing cap ~2cm
            ('bottle', False):   {'partial': 0.35, 'full': 0.58},  # H: grabbing body ~4cm
            ('cylinder', True):  {'partial': 0.45, 'full': 0.65},  # V: top circle ~3cm
            ('cylinder', False): {'partial': 0.35, 'full': 0.58},  # H: full diameter ~4cm
            ('box', True):       {'partial': 0.30, 'full': 0.52},  # V: top face ~5cm
            ('box', False):      {'partial': 0.25, 'full': 0.48},  # H: wider side ~6cm
        }

        # Action clients
        self.arm_client = ActionClient(
            self, FollowJointTrajectory,
            '/arm_controller/follow_joint_trajectory',
            callback_group=self.cb_group)
        self.gripper_client = ActionClient(
            self, FollowJointTrajectory,
            '/gripper_controller/follow_joint_trajectory',
            callback_group=self.cb_group)

        # IK service
        self.ik_client = self.create_client(
            GetPositionIK, '/compute_ik', callback_group=self.cb_group)

        # Grasp attach/detach services (from gazebo_grasp_fix plugin)
        self.attach_srv = self.create_client(
            SetBool, '/grasp_attach', callback_group=self.cb_group)
        self.detach_srv = self.create_client(
            SetBool, '/grasp_detach', callback_group=self.cb_group)

        # Subscribers
        self.create_subscription(PoseStamped, '/detected_objects', self.object_cb, 10)
        self.create_subscription(JointState, '/joint_states', self.joint_cb, 10)

        # Publisher
        self.status_pub = self.create_publisher(String, '/pick_reorient/status', 10)

        self.create_timer(3.0, self.main_loop)
        self.get_logger().info('Pick & Reorient Node started')

        self._home_timer = self.create_timer(
            5.0, self._go_home_once, callback_group=self.cb_group)

    def _wait_for_future(self, future, timeout_sec=10.0):
        """Wait for a future without calling spin_until_future_complete."""
        event = threading.Event()
        future.add_done_callback(lambda _: event.set())
        return event.wait(timeout=timeout_sec)

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    def joint_cb(self, msg):
        self.current_joints = msg

    def object_cb(self, msg):
        for obj in self.detected_objects:
            dx = abs(obj.pose.position.x - msg.pose.position.x)
            dy = abs(obj.pose.position.y - msg.pose.position.y)
            if dx < 0.05 and dy < 0.05:
                return
        if len(self.detected_objects) < 6:
            self.detected_objects.append(msg)
            self.get_logger().info(
                f'Queued object at ({msg.pose.position.x:.3f}, '
                f'{msg.pose.position.y:.3f})')

    def _go_home_once(self):
        self._home_timer.cancel()
        # Wait for both controllers to be available before first motion
        self.get_logger().info('Waiting for arm controller...')
        self.arm_client.wait_for_server()
        self.get_logger().info('Waiting for gripper controller...')
        self.gripper_client.wait_for_server()
        self.get_logger().info('Controllers ready')
        self.controllers_ready = True
        self.go_home()

    def go_home(self):
        self.publish_status('Moving to home')
        self.move_arm([0.0, -1.5708, 1.5708, -1.5708, -1.5708, 0.0], 3.0)

    def _sort_objects(self):
        """Sort detected objects: vertical first, horizontal last.
        This avoids knocking over nearby objects during reorientation."""
        def sort_key(obj_pose):
            parts = obj_pose.header.frame_id.split('::')
            is_vertical = (parts[3] == 'V') if len(parts) >= 4 else False
            return 0 if is_vertical else 1
        self.detected_objects.sort(key=sort_key)

    def main_loop(self):
        if not self.controllers_ready:
            return
        if self.is_busy or not self.detected_objects:
            if not self.detected_objects:
                self.get_logger().info(
                    'Waiting for detections...', throttle_duration_sec=10.0)
            return
        self.is_busy = True
        self._sort_objects()
        obj = self.detected_objects.pop(0)
        try:
            self.execute_pick_reorient_place(obj)
        except Exception as e:
            self.get_logger().error(f'Error: {e}')
            # Recovery: wait for controllers to come back
            self.get_logger().info('Recovering — waiting for controllers...')
            self.arm_client.wait_for_server(timeout_sec=15.0)
            self.gripper_client.wait_for_server(timeout_sec=15.0)
            time.sleep(2.0)
            self.go_home()
        self.is_busy = False

    def _parse_metadata(self, obj_pose):
        parts = obj_pose.header.frame_id.split('::')
        color = parts[1] if len(parts) >= 2 else 'unknown'
        shape = parts[2] if len(parts) >= 3 else 'unknown'
        is_vertical = (parts[3] == 'V') if len(parts) >= 4 else False
        return color, shape, is_vertical

    def _extract_yaw(self, orientation):
        """Extract the Z-axis (yaw) rotation from a quaternion."""
        q = orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return float(np.arctan2(siny_cosp, cosy_cosp))

    def _snap_yaw_90(self, yaw):
        """Snap yaw to nearest multiple of 90 degrees."""
        step = np.pi / 2.0
        return float(np.round(yaw / step) * step)

    # ------------------------------------------------------------------
    # Height constants
    # ------------------------------------------------------------------
    TABLE_Z = 0.71          # table top surface (collision center 0.69 + half 0.02)
    CONTAINER_Z = 0.712     # container base plate Z
    SAFE_Z = 1.10           # travel height — high enough to clear objects during reorient swing

    CONTAINER_WALL = {
        'red': 0.08,
        'green': 0.07,
        'blue': 0.06,
    }

    # Robotiq 2F-85: when arm points down, the inner fingertips are 0.030m
    # ABOVE the TCP in world frame (fingertips at z=0.110 from gripper_base,
    # TCP at z=0.140 from gripper_base → fingertips 0.030m closer to object).
    # To center fingers on the object centroid: TCP = centroid_z - 0.030
    FINGER_ABOVE_TCP = 0.030

    def _grasp_z(self, obj_z, shape, is_vertical):
        """
        Compute TCP grasp Z using REAL detected object centroid.
        Applies fingertip offset and clamps above table.
        """

        # TCP so fingertips align with object centroid
        tcp_z = float(obj_z) - self.FINGER_ABOVE_TCP

        # Safety clamp: never go below table + small margin
        min_z = self.TABLE_Z + 0.005
        tcp_z = max(tcp_z, min_z)
        self.get_logger().info(
            f'_grasp_z REAL: detected={obj_z:.3f} '
            f'tcp={tcp_z:.3f} (clamped_min={min_z:.3f})')

        return tcp_z

    def _insert_z(self, color):
        """TCP Z for placement — ABOVE container walls, not inside."""
        wall_h = self.CONTAINER_WALL.get(color, 0.08)
        return self.CONTAINER_Z + wall_h + 0.10

    # ------------------------------------------------------------------
    # Interpolated Z descent
    # ------------------------------------------------------------------
    def _descend_to(self, x, y, z_start, z_end, yaw=0.0, steps=8,
                    pitch=np.pi - 0.05, roll=0.0):
        """
        Move in small Z steps from z_start to z_end, keeping XY fixed.
        Returns True if all steps succeeded, False on first failure.
        """
        for z in np.linspace(z_start, z_end, steps):
            if not self.move_to_pose(
                    self.make_pose(x, y, float(z), pitch=pitch, yaw=yaw, roll=roll),
                    duration=1.0):
                return False
        return True

    # ------------------------------------------------------------------
    # Main pick-reorient-place
    # ------------------------------------------------------------------
    def execute_pick_reorient_place(self, obj_pose):
        px = obj_pose.pose.position.x
        py = obj_pose.pose.position.y
        obj_z = obj_pose.pose.position.z
        color, shape, is_vertical = self._parse_metadata(obj_pose)

        # Extract object yaw so gripper approaches aligned with the object
        obj_yaw = self._extract_yaw(obj_pose.pose.orientation)

        carry_yaw = obj_yaw
        carry_pitch = np.pi - 0.05

        # Find slot
        slots = self.color_slots.get(color)
        if not slots:
            self.get_logger().error(f'No slots for {color}')
            return
        slot = next((s for s in slots if not s['used']), None)
        if not slot:
            self.get_logger().error(f'All {color} slots full')
            return

        grasp_z = self._grasp_z(obj_z, shape, is_vertical)

        self.publish_status(
            f'Picking {color} {shape} ({"V" if is_vertical else "H"}) '
            f'at ({px:.2f}, {py:.2f}) grasp_z={grasp_z:.3f} '
            f'yaw={np.degrees(obj_yaw):.1f}°')

        # 1. Open gripper
        self.control_gripper(open_gripper=True)

        # 2. Move above object at LOCAL safe height (not always SAFE_Z)
        approach_z_pick = grasp_z + 0.15
        approach_z_pick = max(approach_z_pick, self.TABLE_Z + 0.08)
        approach_z_pick = min(approach_z_pick, self.SAFE_Z)

        self.publish_status('Moving to safe height above object')
        if not self.move_to_pose(self.make_pose(px, py, approach_z_pick, yaw=obj_yaw), duration=2.0):
            # fallback: less constrained pitch
            if not self.move_to_pose(self.make_pose(px, py, approach_z_pick, pitch=2.8, yaw=obj_yaw), duration=2.0):
                self.get_logger().error('Safe-height approach IK failed')
                return

        # 3. Descend — use more steps for precision
        self.publish_status(f'Descending to grasp z={grasp_z:.3f}')
        if not self._descend_to(px, py, approach_z_pick, grasp_z, yaw=obj_yaw, steps=6):
            self.get_logger().error('Descent to grasp failed')
            return

        # 4. Grasp: attach FIRST (while gripper is open around the object),
        #    then gently close. This prevents the object from being launched.
        grip = self.grip_values.get((shape, is_vertical), {'partial': 0.40, 'full': 0.60})
        time.sleep(0.5)
        self.grasp_attach()
        time.sleep(0.3)
        self.control_gripper_partial(grip['partial'])
        time.sleep(1.5)
        self.control_gripper_partial(grip['full'])

        # 5. Lift straight up to SAFE_Z — always clear all objects before moving
        self.publish_status('Lifting to safe height')
        if not self._descend_to(px, py, grasp_z, self.SAFE_Z, yaw=obj_yaw, steps=3):
            self.get_logger().error('Lift failed')
            return

        # 6. If horizontal bottle/cylinder, reorient via IK to tilt gripper 90°
        carry_roll = 0.0
        if not is_vertical and shape in ('bottle', 'cylinder'):
            self.publish_status('Reorienting (H->V) via IK')
            carry_yaw = self._snap_yaw_90(obj_yaw)
            carry_roll = np.pi / 2  # tilt gripper 90° so horizontal object hangs vertical

            # Reorient in place at current XY, high Z
            reorient_z = self.SAFE_Z + 0.05
            if not self.move_to_pose(
                self.make_pose(px, py, reorient_z, pitch=carry_pitch, yaw=carry_yaw, roll=carry_roll),
                duration=3.0
            ):
                # Try opposite roll direction
                carry_roll = -np.pi / 2
                if not self.move_to_pose(
                    self.make_pose(px, py, reorient_z, pitch=carry_pitch, yaw=carry_yaw, roll=carry_roll),
                    duration=3.0
                ):
                    self.get_logger().error('Reorient IK failed')
                    return

        # 7. Travel above container at safe height
        self.publish_status(f'Moving above {color} container')
        if not self.move_to_pose(
            self.make_pose(slot['x'], slot['y'], self.SAFE_Z,
                           pitch=carry_pitch, yaw=carry_yaw, roll=carry_roll),
            duration=3.0
        ):
            self.get_logger().error('Move above container failed')
            return

        # 8. Descend above container and release
        insert_z = self._insert_z(color)
        self.publish_status(f'Descending above container z={insert_z:.3f}')
        if not self._descend_to(slot['x'], slot['y'], self.SAFE_Z, insert_z,
                                yaw=carry_yaw, pitch=carry_pitch, roll=carry_roll, steps=3):
            self.get_logger().error('Descent above container failed')
            return

        # 9. Detach + open gripper
        self.grasp_detach()
        time.sleep(0.3)
        self.control_gripper(open_gripper=True)
        time.sleep(0.5)

        # 10. Ascend back to safe height
        self.move_to_pose(
            self.make_pose(slot['x'], slot['y'], self.SAFE_Z,
                           pitch=carry_pitch, yaw=carry_yaw, roll=carry_roll),

            duration=2.0
        )

        slot['used'] = True
        self.go_home()
        self.publish_status(f'Done: {color} {shape}')

    # ------------------------------------------------------------------
    # Reorientation: rotate a horizontal object to vertical
    # ------------------------------------------------------------------
    def _reorient_horizontal_to_vertical(self, tilt=np.pi/2):

        if self.current_joints is None:
            self.get_logger().warn('No joint state for reorientation')
            return False

        jnames = list(self.current_joints.name)
        jpos   = list(self.current_joints.position)

        def get_joint(name):
            try:
                return jpos[jnames.index(name)]
            except ValueError:
                return 0.0

        pan   = get_joint('shoulder_pan_joint')
        lift  = get_joint('shoulder_lift_joint')
        elbow = get_joint('elbow_joint')
        w1    = get_joint('wrist_1_joint')
        w2    = get_joint('wrist_2_joint')
        w3    = get_joint('wrist_3_joint')

        # Tilt wrist_1 to rotate the gripper 90° so a horizontal object becomes vertical
        w1_tilted = w1 - tilt

        ok = self.move_arm([pan, lift, elbow, w1_tilted, w2, w3], duration=2.5)

        return bool(ok)

    # ------------------------------------------------------------------
    # Grasp attach / detach
    # ------------------------------------------------------------------
    def grasp_attach(self):
        if not self.attach_srv.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn('grasp_attach service not available')
            return
        req = SetBool.Request()
        req.data = True
        try:
            result = self.attach_srv.call(req)
            if result:
                self.get_logger().info(f'Attach: {result.message}')
        except Exception as e:
            self.get_logger().error(f'Attach call failed: {e}')

    def grasp_detach(self):
        if not self.detach_srv.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn('grasp_detach service not available')
            return
        req = SetBool.Request()
        req.data = True
        try:
            result = self.detach_srv.call(req)
            if result:
                self.get_logger().info(f'Detach: {result.message}')
        except Exception as e:
            self.get_logger().error(f'Detach call failed: {e}')

    # ------------------------------------------------------------------
    # Pose creation
    # ------------------------------------------------------------------
    def make_pose(self, x, y, z, pitch=np.pi - 0.05, yaw=0.0, roll=0.0):
        """
        pitch=pi → gripper down, yaw rotates around Z.
        roll tilts the gripper sideways (for reoriented objects).
        Euler order: XYZ intrinsic (pitch around X, roll around Y, yaw around Z).
        """
        pose = PoseStamped()
        pose.header.frame_id = 'world'
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position = Point(x=x, y=y, z=z)
        q = R.from_euler('xyz', [pitch, roll, yaw]).as_quat()
        pose.pose.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
        return pose

    # ------------------------------------------------------------------
    # Motion
    # ------------------------------------------------------------------
    def move_to_pose(self, target, duration=4.0):
        if not self.ik_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().error('IK service not available')
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
            # fallback come prima
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
            self.get_logger().error(f'IK call exception: {e}')
            return False

        if result is None:
            self.get_logger().error('IK call returned None (service error)')
            return False

        if result.error_code.val != MoveItErrorCodes.SUCCESS:
            p = target.pose.position
            self.get_logger().error(
                f'IK error {result.error_code.val} for '
                f'({p.x:.3f}, {p.y:.3f}, {p.z:.3f})')
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
            self.get_logger().error('Arm controller not available')
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
            self.get_logger().error('Arm send_goal timeout')
            return False
        gh = future.result()
        if not gh or not gh.accepted:
            self.get_logger().error('Arm goal rejected')
            return False
        rf = gh.get_result_async()
        if not self._wait_for_future(rf, timeout_sec=duration + 20.0):
            self.get_logger().error('Arm result timeout — cancelling goal')
            try:
                gh.cancel_goal_async()
            except Exception:
                pass
            time.sleep(2.0)
            return False
        res = rf.result()
        if res is None:
            self.get_logger().error("Arm result None")
            return False
        if res.result.error_code != 0:
            self.get_logger().error(f"Arm execution failed, error_code={res.result.error_code}")
            return False
        self.last_arm_seed = list(positions)
        return True

    def control_gripper(self, open_gripper=True):
        if not self.gripper_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('Gripper controller not available')
            return False
        traj = JointTrajectory()
        traj.joint_names = self.gripper_joints
        val = 0.0 if open_gripper else 0.8
        pt = JointTrajectoryPoint()
        pt.positions = [val, val, val, val, -val, -val]
        pt.velocities = [0.0] * 6
        pt.time_from_start = Duration(sec=2, nanosec=0)
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

    def control_gripper_partial(self, value):
        if not self.gripper_client.wait_for_server(timeout_sec=5.0):
            return False

        traj = JointTrajectory()
        traj.joint_names = self.gripper_joints

        pt = JointTrajectoryPoint()
        pt.positions = [value, value, value, value, -value, -value]
        pt.velocities = [0.0] * 6
        pt.time_from_start = Duration(sec=3)

        traj.points.append(pt)

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj

        future = self.gripper_client.send_goal_async(goal)
        self._wait_for_future(future, timeout_sec=10.0)
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
