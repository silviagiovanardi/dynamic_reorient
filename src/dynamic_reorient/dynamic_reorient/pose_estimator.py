import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from cv_bridge import CvBridge
from scipy.spatial.transform import Rotation as R

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped, Point, Quaternion
from visualization_msgs.msg import Marker, MarkerArray
import tf2_ros
from tf2_ros import TransformException


class PoseEstimator(Node):

    TABLE_Z_MIN = 0.70
    TABLE_Z_MAX = 0.95

    # Minimum and maximum contour area to consider
    CONTOUR_AREA_MIN = 150
    CONTOUR_AREA_MAX = 15000

    # Container positions to exclude from detection (x, y)
    # These are the drop-off bins — we don't want to pick them up
    CONTAINER_POSITIONS = [
        (0.55, 0.25), (0.65, 0.25),    # red containers
        (0.55, 0.0),  (0.65, 0.0),     # green containers
        (0.55, -0.25), (0.65, -0.25),  # blue containers
    ]
    CONTAINER_EXCLUSION_RADIUS = 0.08  # meters

    # Depth range in meters
    DEPTH_MIN = 0.1
    DEPTH_MAX = 3.0

    def __init__(self):
        super().__init__('pose_estimator')

        self.bridge = CvBridge()
        self.camera_matrix = None
        self.color_image = None
        self.depth_image = None

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # HSV color ranges for each object class
        self.colors = {
            'red': {
                'lower1': np.array([0, 100, 60]),
                'upper1': np.array([12, 255, 255]),
                'lower2': np.array([165, 100, 60]),
                'upper2': np.array([180, 255, 255]),
            },
            'green': {
                'lower1': np.array([35, 80, 60]),
                'upper1': np.array([90, 255, 255]),
            },
            'blue': {
                'lower1': np.array([90, 70, 60]),
                'upper1': np.array([130, 255, 255]),
            },
        }

        # Map each color to the expected object type
        self.color_to_shape = {
            'red': 'bottle',
            'green': 'box',
            'blue': 'cylinder',
        }

        self.color_to_bin = {
            'red': 'plastic',
            'green': 'paper',
            'blue': 'glass',
        }

        # Subscribers
        self.create_subscription(Image, '/camera/image_raw', self._color_cb, 10)
        self.create_subscription(Image, '/camera/depth/image_raw', self._depth_cb, 10)
        self.create_subscription(CameraInfo, '/camera/camera_info', self._info_cb, 10)

        # Publishers
        self.pose_pub = self.create_publisher(PoseStamped, '/detected_objects', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/detected_markers', 10)
        self.debug_pub = self.create_publisher(Image, '/pose_estimator/debug', 10)

        self.create_timer(0.5, self._detect)

        self.get_logger().info('Pose Estimator started (with shape detection)')

    def _color_cb(self, msg):
        try:
            self.color_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f'Color image error: {e}')

    def _depth_cb(self, msg):
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, 'passthrough')
        except Exception as e:
            self.get_logger().error(f'Depth image error: {e}')

    def _info_cb(self, msg):
        self.camera_matrix = np.array(msg.k).reshape(3, 3)

    def _build_mask(self, hsv, color_name):
        cfg = self.colors[color_name]
        mask = cv2.inRange(hsv, cfg['lower1'], cfg['upper1'])
        if 'lower2' in cfg:
            mask2 = cv2.inRange(hsv, cfg['lower2'], cfg['upper2'])
            mask = cv2.bitwise_or(mask, mask2)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask

    def _sample_depth(self, cy, cx, patch_size=11):
        h, w = self.depth_image.shape[:2]
        half = patch_size // 2
        y0 = max(0, cy - half)
        y1 = min(h, cy + half + 1)
        x0 = max(0, cx - half)
        x1 = min(w, cx + half + 1)

        patch = self.depth_image[y0:y1, x0:x1].copy().astype(np.float64)

        # Depth in mm -> meters
        if self.depth_image.dtype != np.float32:
            patch = patch / 1000.0

        valid = patch[(patch > self.DEPTH_MIN) & (patch < self.DEPTH_MAX)]
        if len(valid) == 0:
            return -1.0

        if float(np.std(valid)) > 0.01:
            return -1.0

        return float(np.median(valid))

    def _classify_shape(self, contour, expected_shape):
        if len(contour) < 5:
            return None

        rect = cv2.minAreaRect(contour)
        w_rect, h_rect = rect[1]
        if w_rect == 0 or h_rect == 0:
            return None

        aspect = max(w_rect, h_rect) / min(w_rect, h_rect)

        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        cnt_area = cv2.contourArea(contour)
        solidity = cnt_area / hull_area if hull_area > 0 else 0

        perimeter = cv2.arcLength(contour, True)
        circularity = (4.0 * np.pi * cnt_area / (perimeter * perimeter)) if perimeter > 0 else 0

        is_vertical = False
        detected_shape = expected_shape

        if expected_shape == 'box':
            if solidity < 0.80:
                return None
            is_vertical = aspect < 1.5

        elif expected_shape == 'cylinder':
            if aspect < 1.4 and circularity > 0.55:
                is_vertical = True
            elif aspect >= 1.4:
                is_vertical = False
            else:
                is_vertical = True

        elif expected_shape == 'bottle':
            if aspect >= 1.6:
                is_vertical = False
            else:
                is_vertical = True

        angle_deg = rect[2]
        if w_rect < h_rect:
            angle_deg += 90.0
        image_angle_rad = np.radians(angle_deg)

        return (detected_shape, image_angle_rad, is_vertical)

    def _detect(self):
        if self.color_image is None or self.depth_image is None or self.camera_matrix is None:
            missing = []
            if self.color_image is None:
                missing.append('color')
            if self.depth_image is None:
                missing.append('depth')
            if self.camera_matrix is None:
                missing.append('camera_info')
            self.get_logger().warn(
                f'Waiting for sensor data: missing {", ".join(missing)}',
                throttle_duration_sec=5.0)
            return

        hsv = cv2.cvtColor(self.color_image, cv2.COLOR_BGR2HSV)
        debug = self.color_image.copy()
        markers = MarkerArray()
        marker_id = 0

        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        cx_cam = self.camera_matrix[0, 2]
        cy_cam = self.camera_matrix[1, 2]

        for color_name in self.colors:
            mask = self._build_mask(hsv, color_name)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < self.CONTOUR_AREA_MIN or area > self.CONTOUR_AREA_MAX:
                    continue

                # M = cv2.moments(cnt)
                # if M['m00'] == 0:
                #     continue

                rect = cv2.minAreaRect(cnt)
                (cx_px, cy_px) = rect[0]
                cx_px = int(cx_px)
                cy_px = int(cy_px)

                if cy_px >= self.depth_image.shape[0] or cx_px >= self.depth_image.shape[1]:
                    continue

                depth = self._sample_depth(cy_px, cx_px)
                if depth < self.DEPTH_MIN or depth > self.DEPTH_MAX:
                    continue

                x_cam = (cx_px - cx_cam) * depth / fx
                y_cam = (cy_px - cy_cam) * depth / fy
                z_cam = depth

                try:
                    tf = self.tf_buffer.lookup_transform(
                        'world', 'camera_optical_frame',
                        rclpy.time.Time(),
                        timeout=rclpy.duration.Duration(seconds=0.1))

                    t = tf.transform.translation
                    q = tf.transform.rotation
                    rot = R.from_quat([q.x, q.y, q.z, q.w])

                    pt_cam = np.array([x_cam, y_cam, z_cam])
                    pt_world = rot.apply(pt_cam) + np.array([t.x, t.y, t.z])

                    if pt_world[2] < self.TABLE_Z_MIN or pt_world[2] > self.TABLE_Z_MAX:
                        continue

                    near_container = False
                    for cx_cont, cy_cont in self.CONTAINER_POSITIONS:
                        dist = np.sqrt((pt_world[0] - cx_cont)**2 + (pt_world[1] - cy_cont)**2)
                        if dist < self.CONTAINER_EXCLUSION_RADIUS:
                            near_container = True
                            break
                    if near_container:
                        continue

                    expected_shape = self.color_to_shape[color_name]
                    result = self._classify_shape(cnt, expected_shape)
                    if result is None:
                        continue

                    detected_shape, image_angle_rad, is_vertical = result

                    dir_cam = np.array([
                        np.cos(image_angle_rad),
                        np.sin(image_angle_rad),
                        0.0
                    ])
                    dir_world = rot.apply(dir_cam)
                    world_yaw = float(np.arctan2(dir_world[1], dir_world[0]))

                    # Build orientation quaternion in world frame
                    if is_vertical:
                        obj_rot = R.from_euler('xyz', [0.0, 0.0, world_yaw])
                    else:
                        obj_rot = R.from_euler('xyz', [np.pi / 2.0, 0.0, world_yaw])
                    oq = obj_rot.as_quat()
                    orientation = Quaternion(
                        x=float(oq[0]), y=float(oq[1]),
                        z=float(oq[2]), w=float(oq[3]))

                    self.get_logger().debug(
                        f'  → {color_name} {detected_shape} '
                        f'{"V" if is_vertical else "H"} '
                        f'at ({pt_world[0]:.3f}, {pt_world[1]:.3f}, {pt_world[2]:.3f}) '
                        f'yaw={np.degrees(world_yaw):.1f}°')

                    # Publish pose
                    pose = PoseStamped()
                    orient_tag = 'V' if is_vertical else 'H'
                    pose.header.frame_id = f'world::{color_name}::{detected_shape}::{orient_tag}'
                    pose.header.stamp = self.get_clock().now().to_msg()
                    pose.pose.position = Point(
                        x=float(pt_world[0]),
                        y=float(pt_world[1]),
                        z=float(pt_world[2]))
                    pose.pose.orientation = orientation
                    self.pose_pub.publish(pose)

                    # Visualization marker
                    marker = Marker()
                    marker.header = pose.header
                    marker.ns = f'{color_name}_{detected_shape}'
                    marker.id = marker_id

                    if detected_shape == 'box':
                        marker.type = Marker.CUBE
                        marker.scale.x = 0.05
                        marker.scale.y = 0.05
                        marker.scale.z = 0.10
                    elif detected_shape == 'bottle':
                        marker.type = Marker.CYLINDER
                        marker.scale.x = 0.05
                        marker.scale.y = 0.05
                        marker.scale.z = 0.14
                    else:
                        marker.type = Marker.CYLINDER
                        marker.scale.x = 0.04
                        marker.scale.y = 0.04
                        marker.scale.z = 0.10

                    marker.action = Marker.ADD
                    marker.pose = pose.pose
                    marker.color.a = 0.8
                    if color_name == 'red':
                        marker.color.r = 1.0
                    elif color_name == 'green':
                        marker.color.g = 1.0
                    else:
                        marker.color.b = 1.0
                    marker.lifetime.sec = 1
                    markers.markers.append(marker)
                    marker_id += 1

                    # Draw debug overlay
                    cv2.drawContours(debug, [cnt], -1, (0, 255, 0), 2)
                    cv2.circle(debug, (cx_px, cy_px), 5, (0, 0, 255), -1)
                    orient_tag = 'V' if is_vertical else 'H'
                    label = f'{color_name}[{detected_shape[0].upper()}/{orient_tag}]'
                    cv2.putText(debug, label,
                                (cx_px + 10, cy_px - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    cv2.putText(debug, f'z={pt_world[2]:.2f}',
                                (cx_px + 10, cy_px + 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

                    self.get_logger().info(
                        f'Detected {color_name} {detected_shape} '
                        f'{"VERTICAL" if is_vertical else "HORIZONTAL"} '
                        f'at ({pt_world[0]:.3f}, {pt_world[1]:.3f}, {pt_world[2]:.3f})')

                except TransformException as e:
                    self.get_logger().warn(f'TF error: {e}')

        self.marker_pub.publish(markers)

        try:
            self.debug_pub.publish(self.bridge.cv2_to_imgmsg(debug, 'bgr8'))
        except Exception:
            pass


def main(args=None):
    rclpy.init(args=args)
    node = PoseEstimator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
