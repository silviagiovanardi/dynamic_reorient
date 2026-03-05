#!/usr/bin/env python3
"""
Utility functions for Dynamic Pick & Reorient.
"""
import numpy as np
from scipy.spatial.transform import Rotation as R
from geometry_msgs.msg import Pose, Point, Quaternion, PoseStamped


def pose_to_matrix(pose: Pose) -> np.ndarray:
    """Convert geometry_msgs Pose to 4x4 homogeneous matrix."""
    mat = np.eye(4)
    mat[:3, :3] = R.from_quat([
        pose.orientation.x, pose.orientation.y,
        pose.orientation.z, pose.orientation.w
    ]).as_matrix()
    mat[:3, 3] = [pose.position.x, pose.position.y, pose.position.z]
    return mat


def matrix_to_pose(mat: np.ndarray) -> Pose:
    """Convert 4x4 homogeneous matrix to geometry_msgs Pose."""
    pose = Pose()
    pose.position = Point(x=mat[0, 3], y=mat[1, 3], z=mat[2, 3])
    q = R.from_matrix(mat[:3, :3]).as_quat()
    pose.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
    return pose


def create_pose_stamped(x, y, z, roll=0, pitch=0, yaw=0, frame='world') -> PoseStamped:
    """Create a PoseStamped from position and Euler angles."""
    ps = PoseStamped()
    ps.header.frame_id = frame
    ps.pose.position = Point(x=x, y=y, z=z)
    q = R.from_euler('xyz', [roll, pitch, yaw]).as_quat()
    ps.pose.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
    return ps


def interpolate_poses(start: Pose, end: Pose, num_steps: int) -> list:
    """Generate interpolated poses between start and end."""
    poses = []
    
    start_pos = np.array([start.position.x, start.position.y, start.position.z])
    end_pos = np.array([end.position.x, end.position.y, end.position.z])
    
    start_q = np.array([start.orientation.x, start.orientation.y,
                        start.orientation.z, start.orientation.w])
    end_q = np.array([end.orientation.x, end.orientation.y,
                      end.orientation.z, end.orientation.w])
    
    for i in range(num_steps + 1):
        t = i / num_steps
        
        # Linear position interpolation
        pos = start_pos + t * (end_pos - start_pos)
        
        # SLERP for orientation
        q = slerp(start_q, end_q, t)
        
        pose = Pose()
        pose.position = Point(x=pos[0], y=pos[1], z=pos[2])
        pose.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
        poses.append(pose)
    
    return poses


def slerp(q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation between quaternions."""
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    
    dot = np.dot(q1, q2)
    
    if dot < 0:
        q2 = -q2
        dot = -dot
    
    if dot > 0.9995:
        result = q1 + t * (q2 - q1)
        return result / np.linalg.norm(result)
    
    theta_0 = np.arccos(dot)
    theta = theta_0 * t
    
    q2_perp = q2 - q1 * dot
    q2_perp = q2_perp / np.linalg.norm(q2_perp)
    
    return q1 * np.cos(theta) + q2_perp * np.sin(theta)


def gripper_down_quaternion() -> Quaternion:
    """Return quaternion for gripper pointing down (Z pointing down)."""
    q = R.from_euler('xyz', [np.pi, 0, 0]).as_quat()
    return Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])


def distance_3d(p1, p2) -> float:
    """Calculate 3D Euclidean distance between two points."""
    return np.sqrt(
        (p1.x - p2.x)**2 +
        (p1.y - p2.y)**2 +
        (p1.z - p2.z)**2
    )