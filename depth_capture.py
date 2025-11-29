#!/usr/bin/env python3
"""
Depth Capture Script - Script 1
Captures images from either OpenCV camera or ROS2 topic and performs depth estimation.
"""

import argparse
import cv2
import numpy as np
import torch
import threading
import time
from collections import deque
from pathlib import Path
from typing import Optional, Tuple

from depth_anything_3.api import DepthAnything3


class DepthCapture:
    """Main class for capturing and processing depth from camera or ROS2."""

    def __init__(
        self,
        model_name: str = 'depth-anything/DA3-SMALL',#"depth-anything/DA3NESTED-GIANT-LARGE",
        process_res: int = 504,
        buffer_size: int = 30,
        use_ros2: bool = False,
        ros2_topic: Optional[str] = None,
        camera_id: int = 0,
    ):
        """
        Initialize the depth capture system.

        Args:
            model_name: Pretrained model name or path
            process_res: Processing resolution for the model
            buffer_size: Number of frames to keep in buffer
            use_ros2: Whether to use ROS2 instead of OpenCV camera
            ros2_topic: ROS2 topic name (required if use_ros2=True)
            camera_id: OpenCV camera device ID (default: 0)
        """
        self.model_name = model_name
        self.process_res = process_res
        self.buffer_size = buffer_size
        self.use_ros2 = use_ros2
        self.ros2_topic = ros2_topic
        self.camera_id = camera_id

        # Image and depth buffers
        self.image_buffer = deque(maxlen=buffer_size)
        self.depth_buffer = deque(maxlen=buffer_size)
        self.conf_buffer = deque(maxlen=buffer_size)
        self.extrinsics_buffer = deque(maxlen=buffer_size)
        self.intrinsics_buffer = deque(maxlen=buffer_size)
        self.timestamp_buffer = deque(maxlen=buffer_size)

        # Latest data (thread-safe access)
        self._lock = threading.Lock()
        self._latest_image = None
        self._latest_depth = None
        self._latest_conf = None
        self._latest_extrinsics = None
        self._latest_intrinsics = None

        # Control flags
        self.running = False
        self.model_loaded = False

        # Initialize model
        print(f"Loading model: {model_name}...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DepthAnything3.from_pretrained(model_name)
        self.model = self.model.to(device=device)
        self.model.eval()
        self.device = device
        self.model_loaded = True
        print(f"Model loaded successfully on {device}")

        # Initialize image source
        if self.use_ros2:
            self._init_ros2()
        else:
            self._init_opencv_camera()

    def _init_opencv_camera(self):
        """Initialize OpenCV camera capture."""
        print(f"Initializing OpenCV camera (ID: {self.camera_id})...")
        self.cap = cv2.VideoCapture(self.camera_id)

        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera with ID {self.camera_id}")

        # Set camera properties for better performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        print("OpenCV camera initialized successfully")

    def _init_ros2(self):
        """Initialize ROS2 subscriber."""
        try:
            import rclpy
            from rclpy.node import Node
            from sensor_msgs.msg import Image
            from cv_bridge import CvBridge

            class ImageSubscriber(Node):
                def __init__(self, topic, callback):
                    super().__init__('depth_capture_subscriber')
                    self.subscription = self.create_subscription(
                        Image,
                        topic,
                        callback,
                        10
                    )
                    self.bridge = CvBridge()

            def image_callback(msg):
                try:
                    cv_image = self.ros_node.bridge.imgmsg_to_cv2(msg, "bgr8")
                    with self._lock:
                        self._latest_ros_image = cv_image
                except Exception as e:
                    print(f"Error converting ROS image: {e}")

            rclpy.init()
            self.ros_node = ImageSubscriber(self.ros2_topic, image_callback)
            self._latest_ros_image = None

            # Start ROS2 spinning in separate thread
            self.ros_thread = threading.Thread(target=self._ros_spin, daemon=True)
            self.ros_thread.start()

            print(f"ROS2 subscriber initialized for topic: {self.ros2_topic}")

        except ImportError:
            raise RuntimeError(
                "ROS2 dependencies not found. Please install: "
                "pip install rclpy sensor_msgs cv_bridge"
            )

    def _ros_spin(self):
        """Spin ROS2 node in separate thread."""
        import rclpy
        rclpy.spin(self.ros_node)

    def _get_frame(self) -> Optional[np.ndarray]:
        """
        Get a frame from the current image source.

        Returns:
            Frame as numpy array (BGR format) or None if unavailable
        """
        if self.use_ros2:
            with self._lock:
                return self._latest_ros_image.copy() if self._latest_ros_image is not None else None
        else:
            ret, frame = self.cap.read()
            return frame if ret else None

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Process a single frame through the depth estimation model.

        Args:
            frame: Input frame (BGR format)

        Returns:
            Tuple of (depth, confidence, extrinsics, intrinsics)
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run inference
        prediction = self.model.inference(
            image=[rgb_frame],
            process_res=self.process_res,
            process_res_method="lower_bound_resize",
        )

        # Extract results
        depth = prediction.depth[0]  # Shape: (H, W)
        conf = prediction.conf[0] if hasattr(prediction, 'conf') and prediction.conf is not None else np.ones_like(depth)

        # Extract camera parameters if available (only if not None)
        extrinsics = prediction.extrinsics[0] if hasattr(prediction, 'extrinsics') and prediction.extrinsics is not None else None
        intrinsics = prediction.intrinsics[0] if hasattr(prediction, 'intrinsics') and prediction.intrinsics is not None else None

        return depth, conf, extrinsics, intrinsics

    def run(self, display: bool = True, save_dir: Optional[str] = None):
        """
        Main loop for capturing and processing frames.

        Args:
            display: Whether to display the output
            save_dir: Optional directory to save results
        """
        self.running = True

        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)

        frame_count = 0
        fps_counter = 0
        fps_start_time = time.time()

        print("Starting capture loop... Press 'q' to quit")

        try:
            while self.running:
                # Get frame
                frame = self._get_frame()

                if frame is None:
                    time.sleep(0.01)
                    continue

                # Process frame
                try:
                    depth, conf, extrinsics, intrinsics = self.process_frame(frame)

                    # Update buffers
                    timestamp = time.time()
                    self.image_buffer.append(frame.copy())
                    self.depth_buffer.append(depth)
                    self.conf_buffer.append(conf)
                    self.extrinsics_buffer.append(extrinsics)
                    self.intrinsics_buffer.append(intrinsics)
                    self.timestamp_buffer.append(timestamp)

                    # Update latest data (thread-safe)
                    with self._lock:
                        self._latest_image = frame.copy()
                        self._latest_depth = depth
                        self._latest_conf = conf
                        self._latest_extrinsics = extrinsics
                        self._latest_intrinsics = intrinsics

                    frame_count += 1
                    fps_counter += 1

                    # Calculate FPS
                    if time.time() - fps_start_time >= 1.0:
                        fps = fps_counter / (time.time() - fps_start_time)
                        print(f"FPS: {fps:.2f}")
                        fps_counter = 0
                        fps_start_time = time.time()

                    # Display
                    if display:
                        # Normalize depth for visualization
                        depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
                        depth_colored = cv2.applyColorMap(depth_normalized.astype(np.uint8), cv2.COLORMAP_INFERNO)

                        # Resize for display
                        h, w = frame.shape[:2]
                        depth_colored_resized = cv2.resize(depth_colored, (w, h))

                        # Concatenate original and depth
                        display_image = np.hstack([frame, depth_colored_resized])

                        cv2.imshow('Depth Capture', display_image)

                        # Check for quit
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q'):
                            break
                    
                    try:
                        if intrinsics is not None:
                            # convert to meters and print the closest object distance
                            focal_length = intrinsics[0, 0]
                            depth_meters = depth * focal_length/300
                            closest_object_distance = np.min(depth_meters)
                            print(f"Closest object distance: {closest_object_distance:.2f} meters")
                            furthest_object_distance = np.max(depth_meters)
                            print(f"Furthest object distance: {furthest_object_distance:.2f} meters")
                        else:
                            depth_meters = depth
                            closest_object_distance = np.min(depth_meters)
                            print(f"Closest object distance: {closest_object_distance:.2f} meters")
                            furthest_object_distance = np.max(depth_meters)
                            print(f"Furthest object distance: {furthest_object_distance:.2f} meters")
                    except Exception as e:
                        print(f"Error converting depth to meters: {e}")

                    # Save if requested
                    if save_dir and frame_count % 30 == 0:  # Save every 30 frames
                        frame_path = save_path / f"frame_{frame_count:06d}.png"
                        depth_path = save_path / f"depth_{frame_count:06d}.npz"
                        cv2.imwrite(str(frame_path), frame)
                        np.savez(
                            str(depth_path),
                            depth=depth,
                            conf=conf,
                            extrinsics=extrinsics,
                            intrinsics=intrinsics,
                            timestamp=timestamp
                        )

                except Exception as e:
                    print(f"Error processing frame: {e}")
                    continue

        finally:
            self.stop()

    def stop(self):
        """Stop the capture and cleanup resources."""
        self.running = False

        if not self.use_ros2 and hasattr(self, 'cap'):
            self.cap.release()

        if self.use_ros2:
            import rclpy
            self.ros_node.destroy_node()
            rclpy.shutdown()

        cv2.destroyAllWindows()
        print("Capture stopped")

    def get_latest_data(self):
        """
        Get the latest captured data (thread-safe).

        Returns:
            Dictionary with latest image, depth, confidence, and camera parameters
        """
        with self._lock:
            return {
                'image': self._latest_image.copy() if self._latest_image is not None else None,
                'depth': self._latest_depth.copy() if self._latest_depth is not None else None,
                'conf': self._latest_conf.copy() if self._latest_conf is not None else None,
                'extrinsics': self._latest_extrinsics.copy() if self._latest_extrinsics is not None else None,
                'intrinsics': self._latest_intrinsics.copy() if self._latest_intrinsics is not None else None,
            }

    def get_buffer_data(self, num_frames: int = -1):
        """
        Get buffered data for multi-view processing.

        Args:
            num_frames: Number of recent frames to return (-1 for all)

        Returns:
            Dictionary with lists of images, depths, etc.
        """
        with self._lock:
            if num_frames == -1 or num_frames > len(self.image_buffer):
                num_frames = len(self.image_buffer)

            return {
                'images': list(self.image_buffer)[-num_frames:],
                'depths': list(self.depth_buffer)[-num_frames:],
                'confs': list(self.conf_buffer)[-num_frames:],
                'extrinsics': list(self.extrinsics_buffer)[-num_frames:],
                'intrinsics': list(self.intrinsics_buffer)[-num_frames:],
                'timestamps': list(self.timestamp_buffer)[-num_frames:],
            }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Depth Capture from Camera or ROS2")

    # Source selection
    parser.add_argument(
        '--use-ros2',
        action='store_true',
        help='Use ROS2 topic instead of OpenCV camera'
    )
    parser.add_argument(
        '--ros2-topic',
        type=str,
        default='/camera/image_raw',
        help='ROS2 image topic (default: /camera/image_raw)'
    )
    parser.add_argument(
        '--camera-id',
        type=int,
        default=0,
        help='OpenCV camera device ID (default: 0)'
    )

    # Model configuration
    parser.add_argument(
        '--model-name',
        type=str,
        default='depth-anything/DA3METRIC-LARGE',
        # default='depth-anything/DA3-BASE',
        # default='depth-anything/DA3-SMALL',
        help='Pretrained model name or path'
    )
    parser.add_argument(
        '--process-res',
        type=int,
        default=504,
        help='Processing resolution (default: 504)'
    )

    # Display and saving
    parser.add_argument(
        '--no-display',
        action='store_true',
        help='Disable visualization'
    )
    parser.add_argument(
        '--save-dir',
        type=str,
        help='Directory to save frames and depth data'
    )
    parser.add_argument(
        '--buffer-size',
        type=int,
        default=30,
        help='Number of frames to keep in buffer (default: 30)'
    )

    args = parser.parse_args()

    # Validate ROS2 arguments
    if args.use_ros2 and not args.ros2_topic:
        parser.error("--ros2-topic is required when using --use-ros2")

    # Create and run capture
    capture = DepthCapture(
        model_name=args.model_name,
        process_res=args.process_res,
        buffer_size=args.buffer_size,
        use_ros2=args.use_ros2,
        ros2_topic=args.ros2_topic if args.use_ros2 else None,
        camera_id=args.camera_id,
    )

    capture.run(
        display=not args.no_display,
        save_dir=args.save_dir
    )


if __name__ == "__main__":
    main()
