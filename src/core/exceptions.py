"""
Custom Exception Hierarchy for Face ID System
"""

class FaceIDError(Exception):
    """Base exception for all Face ID errors"""
    pass

class FaceDetectionError(FaceIDError):
    """Raised when face detection fails"""
    pass

class FaceRecognitionError(FaceIDError):
    """Raised when face recognition fails"""
    pass

class DatabaseError(FaceIDError):
    """Raised for database operations failures"""
    pass

class ConfigurationError(FaceIDError):
    """Raised when configuration is invalid or missing"""
    pass

class CameraError(FaceIDError):
    """Raised for camera initialization or read errors"""
    pass

class RegistrationError(FaceIDError):
    """Raised when person registration fails"""
    pass
