"""
3D Human Pose Normalization Script
==================================

This script performs comprehensive 3D human pose normalization from images using
Google's MediaPipe framework. It provides a modular, decoupled design with
step-by-step visualization of the entire normalization process.

Author: Xi Wu
Date: 2025-07-16
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
import cv2
from typing import List, Optional, Dict, Union, Tuple
import os


class PoseNormalizer:
    """
    A comprehensive 3D pose normalization class that handles detection,
    centering, scaling, and rotation normalization of human poses.
    """

    def __init__(self, model_path: str):
        """
        Initialize the PoseNormalizer with MediaPipe pose landmarker.

        Args:
            model_path (str): Path to the MediaPipe pose_landmarker_heavy.task model file
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Initialize MediaPipe pose landmarker
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            output_segmentation_masks=False
        )
        self.landmarker = vision.PoseLandmarker.create_from_options(options)

        # Create joint mapping dictionary for readable code
        self.joint_mapping = self._create_joint_mapping()

        # Define bone connections for visualization
        self.bone_connections = self._define_bone_connections()

    def _create_joint_mapping(self) -> Dict[str, int]:
        """Create a mapping from MediaPipe pose landmark names to indices."""
        pose_landmarks = mp.solutions.pose.PoseLandmark
        return {
            'NOSE': pose_landmarks.NOSE.value,
            'LEFT_EYE_INNER': pose_landmarks.LEFT_EYE_INNER.value,
            'LEFT_EYE': pose_landmarks.LEFT_EYE.value,
            'LEFT_EYE_OUTER': pose_landmarks.LEFT_EYE_OUTER.value,
            'RIGHT_EYE_INNER': pose_landmarks.RIGHT_EYE_INNER.value,
            'RIGHT_EYE': pose_landmarks.RIGHT_EYE.value,
            'RIGHT_EYE_OUTER': pose_landmarks.RIGHT_EYE_OUTER.value,
            'LEFT_EAR': pose_landmarks.LEFT_EAR.value,
            'RIGHT_EAR': pose_landmarks.RIGHT_EAR.value,
            'MOUTH_LEFT': pose_landmarks.MOUTH_LEFT.value,
            'MOUTH_RIGHT': pose_landmarks.MOUTH_RIGHT.value,
            'LEFT_SHOULDER': pose_landmarks.LEFT_SHOULDER.value,
            'RIGHT_SHOULDER': pose_landmarks.RIGHT_SHOULDER.value,
            'LEFT_ELBOW': pose_landmarks.LEFT_ELBOW.value,
            'RIGHT_ELBOW': pose_landmarks.RIGHT_ELBOW.value,
            'LEFT_WRIST': pose_landmarks.LEFT_WRIST.value,
            'RIGHT_WRIST': pose_landmarks.RIGHT_WRIST.value,
            'LEFT_PINKY': pose_landmarks.LEFT_PINKY.value,
            'RIGHT_PINKY': pose_landmarks.RIGHT_PINKY.value,
            'LEFT_INDEX': pose_landmarks.LEFT_INDEX.value,
            'RIGHT_INDEX': pose_landmarks.RIGHT_INDEX.value,
            'LEFT_THUMB': pose_landmarks.LEFT_THUMB.value,
            'RIGHT_THUMB': pose_landmarks.RIGHT_THUMB.value,
            'LEFT_HIP': pose_landmarks.LEFT_HIP.value,
            'RIGHT_HIP': pose_landmarks.RIGHT_HIP.value,
            'LEFT_KNEE': pose_landmarks.LEFT_KNEE.value,
            'RIGHT_KNEE': pose_landmarks.RIGHT_KNEE.value,
            'LEFT_ANKLE': pose_landmarks.LEFT_ANKLE.value,
            'RIGHT_ANKLE': pose_landmarks.RIGHT_ANKLE.value,
            'LEFT_HEEL': pose_landmarks.LEFT_HEEL.value,
            'RIGHT_HEEL': pose_landmarks.RIGHT_HEEL.value,
            'LEFT_FOOT_INDEX': pose_landmarks.LEFT_FOOT_INDEX.value,
            'RIGHT_FOOT_INDEX': pose_landmarks.RIGHT_FOOT_INDEX.value,
        }

    def _define_bone_connections(self) -> List[Tuple[str, str]]:
        """Define bone connections for visualization."""
        return [
            # Face connections
            ('NOSE', 'LEFT_EYE'), ('NOSE', 'RIGHT_EYE'),
            ('LEFT_EYE', 'LEFT_EAR'), ('RIGHT_EYE', 'RIGHT_EAR'),
            ('MOUTH_LEFT', 'MOUTH_RIGHT'),

            # Torso connections
            ('LEFT_SHOULDER', 'RIGHT_SHOULDER'),
            ('LEFT_SHOULDER', 'LEFT_HIP'),
            ('RIGHT_SHOULDER', 'RIGHT_HIP'),
            ('LEFT_HIP', 'RIGHT_HIP'),

            # Left arm
            ('LEFT_SHOULDER', 'LEFT_ELBOW'),
            ('LEFT_ELBOW', 'LEFT_WRIST'),
            ('LEFT_WRIST', 'LEFT_PINKY'),
            ('LEFT_WRIST', 'LEFT_INDEX'),
            ('LEFT_WRIST', 'LEFT_THUMB'),

            # Right arm
            ('RIGHT_SHOULDER', 'RIGHT_ELBOW'),
            ('RIGHT_ELBOW', 'RIGHT_WRIST'),
            ('RIGHT_WRIST', 'RIGHT_PINKY'),
            ('RIGHT_WRIST', 'RIGHT_INDEX'),
            ('RIGHT_WRIST', 'RIGHT_THUMB'),

            # Left leg
            ('LEFT_HIP', 'LEFT_KNEE'),
            ('LEFT_KNEE', 'LEFT_ANKLE'),
            ('LEFT_ANKLE', 'LEFT_HEEL'),
            ('LEFT_ANKLE', 'LEFT_FOOT_INDEX'),

            # Right leg
            ('RIGHT_HIP', 'RIGHT_KNEE'),
            ('RIGHT_KNEE', 'RIGHT_ANKLE'),
            ('RIGHT_ANKLE', 'RIGHT_HEEL'),
            ('RIGHT_ANKLE', 'RIGHT_FOOT_INDEX'),
        ]

    def detect_3d_landmarks(self, image_path: str) -> Optional[np.ndarray]:
        """
        Detect 3D pose landmarks from an input image.

        Args:
            image_path (str): Path to the input image

        Returns:
            Optional[np.ndarray]: (J, 3) array of 3D landmarks or None if no pose detected
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        # Load image
        image = mp.Image.create_from_file(image_path)

        # Detect pose landmarks
        detection_result = self.landmarker.detect(image)

        # Check if pose was detected
        if not detection_result.pose_world_landmarks:
            print(f"No pose detected in image: {image_path}")
            return None

        # Extract 3D world landmarks
        pose_landmarks = detection_result.pose_world_landmarks[0]

        # Convert to NumPy array and invert Y-axis (MediaPipe Y-down to Y-up)
        landmarks = np.array([[lm.x, -lm.y, lm.z] for lm in pose_landmarks])

        return landmarks

    def center_pose(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Center the pose by subtracting the pelvis center.

        Args:
            landmarks (np.ndarray): (J, 3) array of 3D landmarks

        Returns:
            np.ndarray: Centered (J, 3) array of landmarks
        """
        # Calculate pelvis center (midpoint of left and right hip)
        left_hip_idx = self.joint_mapping['LEFT_HIP']
        right_hip_idx = self.joint_mapping['RIGHT_HIP']

        pelvis_center = (landmarks[left_hip_idx] +
                         landmarks[right_hip_idx]) / 2.0

        # Center all landmarks
        centered_landmarks = landmarks - pelvis_center

        return centered_landmarks

    def scale_pose(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Scale the pose to normalize the torso length to 1.0.

        Args:
            landmarks (np.ndarray): (J, 3) array of centered landmarks

        Returns:
            np.ndarray: Scaled (J, 3) array of landmarks
        """
        # Calculate shoulder midpoint
        left_shoulder_idx = self.joint_mapping['LEFT_SHOULDER']
        right_shoulder_idx = self.joint_mapping['RIGHT_SHOULDER']
        shoulder_midpoint = (landmarks[left_shoulder_idx] + landmarks[right_shoulder_idx]) / 2.0

        # Calculate hip midpoint (should be close to origin after centering)
        left_hip_idx = self.joint_mapping['LEFT_HIP']
        right_hip_idx = self.joint_mapping['RIGHT_HIP']
        hip_midpoint = (landmarks[left_hip_idx] + landmarks[right_hip_idx]) / 2.0

        # Calculate torso length
        torso_length = np.linalg.norm(shoulder_midpoint - hip_midpoint)

        if torso_length == 0:
            print("Warning: Torso length is zero, skipping scaling")
            return landmarks

        # Scale to make torso length = 1.0
        scaling_factor = 1.0 / torso_length
        scaled_landmarks = landmarks * scaling_factor

        return scaled_landmarks

    def normalize_rotation(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Normalize the pose rotation to align with world axes.

        Args:
            landmarks (np.ndarray): (J, 3) array of centered and scaled landmarks

        Returns:
            np.ndarray: Rotation-normalized (J, 3) array of landmarks
        """
        # Get key points
        left_shoulder_idx = self.joint_mapping['LEFT_SHOULDER']
        right_shoulder_idx = self.joint_mapping['RIGHT_SHOULDER']
        left_hip_idx = self.joint_mapping['LEFT_HIP']
        right_hip_idx = self.joint_mapping['RIGHT_HIP']

        # Calculate body-local coordinate system
        shoulder_midpoint = (landmarks[left_shoulder_idx] + landmarks[right_shoulder_idx]) / 2.0
        hip_midpoint = (landmarks[left_hip_idx] + landmarks[right_hip_idx]) / 2.0

        # X-axis: along shoulders (left to right)
        x_axis = landmarks[right_shoulder_idx] - landmarks[left_shoulder_idx]
        x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-8)

        # Y-axis: along spine (hip to shoulder midpoint)
        y_axis = shoulder_midpoint - hip_midpoint
        y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-8)

        # Z-axis: cross product of X and Y
        z_axis = np.cross(x_axis, y_axis)
        z_axis = z_axis / (np.linalg.norm(z_axis) + 1e-8)

        # Recompute Y-axis to ensure orthogonality
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-8)

        # Construct rotation matrix (body-to-world)
        R = np.column_stack([x_axis, y_axis, z_axis])

        # Apply inverse rotation (world-to-standard orientation)
        normalized_landmarks = landmarks @ R.T

        return normalized_landmarks

    def process_image(self, image_path: str) -> Optional[Dict]:
        """
        Process a single image through the complete normalization pipeline.

        Args:
            image_path (str): Path to the input image

        Returns:
            Optional[Dict]: Dictionary containing all processing results or None if failed
        """
        # Step 1: Detect 3D landmarks
        original_landmarks = self.detect_3d_landmarks(image_path)
        if original_landmarks is None:
            return None

        # Step 2: Center the pose
        centered_landmarks = self.center_pose(original_landmarks)

        # Step 3: Scale the pose
        scaled_landmarks = self.scale_pose(centered_landmarks)

        # Step 4: Normalize rotation
        normalized_landmarks = self.normalize_rotation(scaled_landmarks)
        # normalized_landmarks = scaled_landmarks
        # Calculate body angles
        body_angles = calculate_body_angles(normalized_landmarks, self.joint_mapping)

        return {
            'original': original_landmarks,
            'centered': centered_landmarks,
            'scaled': scaled_landmarks,
            'normalized': normalized_landmarks,
            'angles': body_angles,
            'image_path': image_path
        }

    def process_images(self, image_paths: Union[str, List[str]]) -> List[Dict]:
        """
        Process single image or list of images.

        Args:
            image_paths: Single image path or list of image paths

        Returns:
            List[Dict]: List of processing results
        """
        if isinstance(image_paths, str):
            image_paths = [image_paths]

        results = []
        for image_path in image_paths:
            result = self.process_image(image_path)
            if result is not None:
                results.append(result)

        return results


def calculate_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """
    Calculate the angle at vertex p2 formed by points p1, p2, p3.

    Args:
        p1, p2, p3 (np.ndarray): 3D points

    Returns:
        float: Angle in degrees
    """
    # Vectors from p2 to p1 and p2 to p3
    v1 = p1 - p2
    v2 = p3 - p2

    # Calculate angle using dot product
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Ensure valid range

    angle_rad = np.arccos(cos_angle)
    angle_deg = np.degrees(angle_rad)

    return angle_deg


def calculate_body_angles(landmarks: np.ndarray, joint_mapping: Dict[str, int]) -> Dict[str, float]:
    """
    Calculate key body angles from normalized landmarks.

    Args:
        landmarks (np.ndarray): (J, 3) array of normalized landmarks
        joint_mapping (Dict[str, int]): Mapping from joint names to indices

    Returns:
        Dict[str, float]: Dictionary of angle names to values in degrees
    """
    angles = {}

    try:
        # Right elbow angle
        right_shoulder = landmarks[joint_mapping['RIGHT_SHOULDER']]
        right_elbow = landmarks[joint_mapping['RIGHT_ELBOW']]
        right_wrist = landmarks[joint_mapping['RIGHT_WRIST']]
        angles['Right Elbow'] = calculate_angle(right_shoulder, right_elbow, right_wrist)

        # Left elbow angle
        left_shoulder = landmarks[joint_mapping['LEFT_SHOULDER']]
        left_elbow = landmarks[joint_mapping['LEFT_ELBOW']]
        left_wrist = landmarks[joint_mapping['LEFT_WRIST']]
        angles['Left Elbow'] = calculate_angle(left_shoulder, left_elbow, left_wrist)

        # Right knee angle
        right_hip = landmarks[joint_mapping['RIGHT_HIP']]
        right_knee = landmarks[joint_mapping['RIGHT_KNEE']]
        right_ankle = landmarks[joint_mapping['RIGHT_ANKLE']]
        angles['Right Knee'] = calculate_angle(right_hip, right_knee, right_ankle)

        # Left knee angle
        left_hip = landmarks[joint_mapping['LEFT_HIP']]
        left_knee = landmarks[joint_mapping['LEFT_KNEE']]
        left_ankle = landmarks[joint_mapping['LEFT_ANKLE']]
        angles['Left Knee'] = calculate_angle(left_hip, left_knee, left_ankle)

        # Shoulder angle (angle between shoulders and spine)
        left_shoulder = landmarks[joint_mapping['LEFT_SHOULDER']]
        right_shoulder = landmarks[joint_mapping['RIGHT_SHOULDER']]
        shoulder_mid = (left_shoulder + right_shoulder) / 2
        left_hip = landmarks[joint_mapping['LEFT_HIP']]
        right_hip = landmarks[joint_mapping['RIGHT_HIP']]
        hip_mid = (left_hip + right_hip) / 2
        angles['Torso Upright'] = calculate_angle(left_shoulder, shoulder_mid, hip_mid)

    except KeyError as e:
        print(f"Warning: Could not calculate angle due to missing joint: {e}")

    return angles


def visualize_pose_3d(landmarks: np.ndarray, bone_connections: List[Tuple[str, str]],
                      joint_mapping: Dict[str, int], ax: plt.Axes, title: str):
    """
    Visualize a single 3D pose on a given axis.

    Args:
        landmarks (np.ndarray): (J, 3) array of 3D landmarks
        bone_connections (List[Tuple[str, str]]): List of bone connections
        joint_mapping (Dict[str, int]): Mapping from joint names to indices
        ax (plt.Axes): Matplotlib 3D axis
        title (str): Title for the plot
    """
    # Plot joints
    ax.scatter(landmarks[:, 0], landmarks[:, 2], landmarks[:, 1],
               c='red', s=50, alpha=0.8)

    # Plot bones
    for joint1_name, joint2_name in bone_connections:
        if joint1_name in joint_mapping and joint2_name in joint_mapping:
            idx1 = joint_mapping[joint1_name]
            idx2 = joint_mapping[joint2_name]

            x_coords = [landmarks[idx1, 0], landmarks[idx2, 0]]
            y_coords = [landmarks[idx1, 1], landmarks[idx2, 1]]
            z_coords = [landmarks[idx1, 2], landmarks[idx2, 2]]

            ax.plot(x_coords, z_coords, y_coords, 'b-', linewidth=2, alpha=0.7)

    # Set equal aspect ratio and labels
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_zlabel('Y')
    ax.set_title(title)

    # Set equal aspect ratio
    max_range = np.array([landmarks[:, 0].max() - landmarks[:, 0].min(),
                          landmarks[:, 1].max() - landmarks[:, 1].min(),
                          landmarks[:, 2].max() - landmarks[:, 2].min()]).max() / 2.0
    mid_x = (landmarks[:, 0].max() + landmarks[:, 0].min()) * 0.5
    mid_z = (landmarks[:, 1].max() + landmarks[:, 1].min()) * 0.5
    mid_y = (landmarks[:, 2].max() + landmarks[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)


def visualize_all_steps(original: np.ndarray, centered: np.ndarray,
                        scaled: np.ndarray, normalized: np.ndarray,
                        bone_connections: List[Tuple[str, str]],
                        joint_mapping: Dict[str, int]):
    """
    Visualize all steps of pose normalization in a 2x2 grid.

    Args:
        original, centered, scaled, normalized (np.ndarray): Pose arrays at each step
        bone_connections (List[Tuple[str, str]]): List of bone connections
        joint_mapping (Dict[str, int]): Mapping from joint names to indices
    """
    fig = plt.figure(figsize=(16, 12))

    # Original pose
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    visualize_pose_3d(original, bone_connections, joint_mapping, ax1, "1. Original 3D Pose")

    # Centered pose
    ax2 = fig.add_subplot(2, 2, 2, projection='3d')
    visualize_pose_3d(centered, bone_connections, joint_mapping, ax2, "2. Centered Pose")

    # Scaled pose
    ax3 = fig.add_subplot(2, 2, 3, projection='3d')
    visualize_pose_3d(scaled, bone_connections, joint_mapping, ax3, "3. Scaled Pose")

    # Normalized pose
    ax4 = fig.add_subplot(2, 2, 4, projection='3d')
    visualize_pose_3d(normalized, bone_connections, joint_mapping, ax4, "4. Rotation Normalized Pose")

    plt.tight_layout()
    return fig


def download_sample_image():
    """Download a sample image for testing (optional helper function)."""
    import urllib.request

    sample_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1f/Standing_man.jpg/256px-Standing_man.jpg"
    sample_path = "sample_pose.jpg"

    if not os.path.exists(sample_path):
        try:
            print(f"Downloading sample image to {sample_path}...")
            urllib.request.urlretrieve(sample_url, sample_path)
            print("Sample image downloaded successfully!")
        except Exception as e:
            print(f"Failed to download sample image: {e}")
            print("Please provide your own image path.")
            return None

    return sample_path


if __name__ == "__main__":
    # Configuration
    MODEL_PATH = "pose_landmarker_heavy.task"  # Path to MediaPipe model

    # Download model if not exists (you need to download this manually)
    if not os.path.exists(MODEL_PATH):
        print(f"""
        Model file '{MODEL_PATH}' not found!

        Please download the MediaPipe Pose Landmarker model:
        1. Go to: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
        2. Download 'pose_landmarker_heavy.task'
        3. Place it in the same directory as this script

        Or use wget:
        wget -O pose_landmarker_heavy.task https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task
        """)
        exit(1)

    # Sample image (you can replace this with your own image path)
    sample_image_path = download_sample_image()
    if sample_image_path is None:
        sample_image_path = input("Please enter the path to your image: ")

    try:
        # Initialize PoseNormalizer
        print("Initializing PoseNormalizer...")
        normalizer = PoseNormalizer(MODEL_PATH)

        # Process the image
        print(f"Processing image: {sample_image_path}")
        result = normalizer.process_image(sample_image_path)

        if result is None:
            print("Failed to process image. Please check if a person is visible in the image.")
            exit(1)

        # Extract results
        original = result['original']
        centered = result['centered']
        scaled = result['scaled']
        normalized = result['normalized']
        angles = result['angles']

        # Print final results
        print("\n" + "=" * 50)
        print("POSE NORMALIZATION RESULTS")
        print("=" * 50)

        print(f"\nFinal normalized 3D coordinates shape: {normalized.shape}")
        print(f"Coordinate range: X[{normalized[:, 0].min():.3f}, {normalized[:, 0].max():.3f}], "
              f"Y[{normalized[:, 1].min():.3f}, {normalized[:, 1].max():.3f}], "
              f"Z[{normalized[:, 2].min():.3f}, {normalized[:, 2].max():.3f}]")

        print("\nKey Body Angles:")
        for angle_name, angle_value in angles.items():
            print(f"  {angle_name}: {angle_value:.1f}°")

        # Visualize all steps
        print("\nGenerating visualization...")
        fig = visualize_all_steps(original, centered, scaled, normalized,
                                  normalizer.bone_connections, normalizer.joint_mapping)

        plt.show()

        print("\nProcessing completed successfully!")

    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback

        traceback.print_exc()
