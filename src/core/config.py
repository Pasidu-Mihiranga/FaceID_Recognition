"""
Centralized Configuration for Face ID System

All thresholds, paths, and settings in one place.
Previously scattered across main.py (0.15), face_id_system.py (0.6),
video_registration_cli.py (0.7), and minimal_face_id.py (0.8).
"""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DetectionConfig:
    """Face detection settings"""
    detector_type: str = 'opencv'
    min_face_size: tuple = (30, 30)
    scale_factor: float = 1.1
    min_neighbors: int = 5
    # Lenient fallback parameters
    lenient_scale_factor: float = 1.01
    lenient_min_neighbors: int = 1
    lenient_min_size: tuple = (10, 10)


@dataclass
class RecognitionConfig:
    """Face recognition settings"""
    model_type: str = 'deepface'
    # Unified recognition threshold (was 0.15/0.6/0.7/0.8 in different files)
    threshold: float = 0.45
    # Adaptive threshold bounds
    threshold_min: float = 0.2
    threshold_max: float = 0.8
    # High quality image adjustment
    high_quality_adjustment: float = -0.05
    # Low quality image adjustment
    low_quality_adjustment: float = 0.10


@dataclass
class LearningConfig:
    """Continuous learning settings"""
    threshold: float = 0.7
    # DBSCAN parameters for outlier removal
    dbscan_eps: float = 0.3
    dbscan_min_samples: int = 2
    # Learning constraints
    max_embeddings_per_person: int = 50
    learning_interval_seconds: int = 300
    min_confidence_for_learning: float = 0.8


@dataclass
class CameraConfig:
    """Camera integration settings"""
    camera_index: int = 0
    frame_width: int = 640
    frame_height: int = 480
    target_fps: int = 30
    # Recognition cooldown (seconds between recognitions)
    recognition_cooldown: float = 0.5
    # Minimum face size for recognition in camera feed
    min_face_size_camera: tuple = (50, 50)


@dataclass
class VideoRegistrationConfig:
    """Video registration settings"""
    video_duration: int = 15  # seconds
    fps: int = 30
    augmentation_factor: int = 4
    min_quality_score: float = 0.5
    target_face_size: tuple = (160, 160)


@dataclass
class DatabaseConfig:
    """Database settings"""
    db_path: str = 'data/face_database.db'
    # Recognition log retention (days)
    log_retention_days: int = 90


@dataclass
class WebConfig:
    """Web interface settings"""
    host: str = '0.0.0.0'
    port: int = 5000
    max_upload_size_mb: int = 16
    upload_folder: str = 'data/uploads'
    allowed_extensions: tuple = ('png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp')


@dataclass
class PathsConfig:
    """File system paths"""
    base_dir: str = field(default_factory=lambda: os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir: str = 'data'
    identity_storage: str = 'data/face_identities'
    thumbnail_dir: str = 'data/thumbnails'
    video_frames_dir: str = 'data/video_frames'
    embeddings_dir: str = 'data/embeddings'
    registered_faces_dir: str = 'data/registered_faces'
    upload_dir: str = 'data/uploads'
    model_dir: str = 'models'


@dataclass
class FaceIDConfig:
    """Master configuration combining all settings"""
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    recognition: RecognitionConfig = field(default_factory=RecognitionConfig)
    learning: LearningConfig = field(default_factory=LearningConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    video_registration: VideoRegistrationConfig = field(default_factory=VideoRegistrationConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    web: WebConfig = field(default_factory=WebConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)

    def ensure_directories(self) -> None:
        """Create all required data directories if they don't exist."""
        dirs = [
            self.paths.data_dir,
            self.paths.identity_storage,
            self.paths.thumbnail_dir,
            self.paths.video_frames_dir,
            self.paths.embeddings_dir,
            self.paths.registered_faces_dir,
            self.paths.upload_dir,
        ]
        for d in dirs:
            os.makedirs(d, exist_ok=True)


# Global default config instance
DEFAULT_CONFIG = FaceIDConfig()
