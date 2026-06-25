#!/usr/bin/env python3
"""
Comprehensive System Cleanup Script
Removes unwanted files, cleans database, and optimizes the system
"""

import os
import sys
import logging
import sqlite3
import glob
import shutil
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SystemCleanup:
    """Comprehensive system cleanup utility"""
    
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.cleanup_stats = {
            'files_deleted': 0,
            'directories_deleted': 0,
            'database_records_cleaned': 0,
            'space_freed_mb': 0
        }
    
    def cleanup_test_files(self):
        """Remove test files and temporary data"""
        logger.info("=== Cleaning Test Files ===")
        
        test_patterns = [
            "test_*.py",
            "test_*.jpg",
            "test_*.png",
            "test_*.webm",
            "debug_*.jpg",
            "debug_*.png",
            "*_temp_*",
            "temp_*",
            "learning_temp_*"
        ]
        
        for pattern in test_patterns:
            files = glob.glob(os.path.join(self.base_dir, pattern))
            for file_path in files:
                try:
                    if os.path.exists(file_path):
                        size = os.path.getsize(file_path)
                        os.remove(file_path)
                        self.cleanup_stats['files_deleted'] += 1
                        self.cleanup_stats['space_freed_mb'] += size / (1024 * 1024)
                        logger.info(f"Deleted test file: {os.path.basename(file_path)}")
                except Exception as e:
                    logger.warning(f"Failed to delete {file_path}: {e}")
    
    def cleanup_pycache(self):
        """Remove all __pycache__ directories"""
        logger.info("=== Cleaning Python Cache ===")
        
        pycache_dirs = []
        for root, dirs, files in os.walk(self.base_dir):
            for dir_name in dirs:
                if dir_name == '__pycache__':
                    pycache_dirs.append(os.path.join(root, dir_name))
        
        for pycache_dir in pycache_dirs:
            try:
                shutil.rmtree(pycache_dir)
                self.cleanup_stats['directories_deleted'] += 1
                logger.info(f"Deleted pycache: {pycache_dir}")
            except Exception as e:
                logger.warning(f"Failed to delete {pycache_dir}: {e}")
    
    def cleanup_old_uploads(self, days_old=7):
        """Remove old upload files"""
        logger.info(f"=== Cleaning Uploads Older Than {days_old} Days ===")
        
        uploads_dir = os.path.join(self.base_dir, "data", "uploads")
        if not os.path.exists(uploads_dir):
            return
        
        cutoff_date = datetime.now() - timedelta(days=days_old)
        
        for file_name in os.listdir(uploads_dir):
            file_path = os.path.join(uploads_dir, file_name)
            if os.path.isfile(file_path):
                try:
                    file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                    if file_time < cutoff_date:
                        size = os.path.getsize(file_path)
                        os.remove(file_path)
                        self.cleanup_stats['files_deleted'] += 1
                        self.cleanup_stats['space_freed_mb'] += size / (1024 * 1024)
                        logger.info(f"Deleted old upload: {file_name}")
                except Exception as e:
                    logger.warning(f"Failed to delete {file_path}: {e}")
    
    def cleanup_orphaned_video_frames(self):
        """Remove video frames for deleted persons"""
        logger.info("=== Cleaning Orphaned Video Frames ===")
        
        video_frames_dir = os.path.join(self.base_dir, "data", "video_frames")
        if not os.path.exists(video_frames_dir):
            return
        
        # Get list of current persons from database
        try:
            db_path = os.path.join(self.base_dir, "data", "face_database.db")
            if os.path.exists(db_path):
                with sqlite3.connect(db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT name FROM persons")
                    current_persons = {row[0].replace(' ', '_') for row in cursor.fetchall()}
            else:
                current_persons = set()
        except Exception as e:
            logger.warning(f"Failed to get current persons: {e}")
            current_persons = set()
        
        # Check video frame directories
        for dir_name in os.listdir(video_frames_dir):
            dir_path = os.path.join(video_frames_dir, dir_name)
            if os.path.isdir(dir_path):
                # Extract person name from directory name
                person_name = dir_name.split('_')[0]  # Get first part before timestamp
                
                if person_name not in current_persons:
                    try:
                        shutil.rmtree(dir_path)
                        self.cleanup_stats['directories_deleted'] += 1
                        logger.info(f"Deleted orphaned video frames: {dir_name}")
                    except Exception as e:
                        logger.warning(f"Failed to delete {dir_path}: {e}")
    
    def cleanup_database(self):
        """Clean up database records and orphaned data"""
        logger.info("=== Cleaning Database ===")
        
        db_path = os.path.join(self.base_dir, "data", "face_database.db")
        if not os.path.exists(db_path):
            logger.info("No database found to clean")
            return
        
        try:
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                
                # Clean up orphaned face images
                cursor.execute("""
                    DELETE FROM face_images 
                    WHERE person_id NOT IN (SELECT id FROM persons)
                """)
                orphaned_images = cursor.rowcount
                self.cleanup_stats['database_records_cleaned'] += orphaned_images
                logger.info(f"Deleted {orphaned_images} orphaned face image records")
                
                # Clean up orphaned recognition logs
                cursor.execute("""
                    DELETE FROM recognition_logs 
                    WHERE person_id NOT IN (SELECT id FROM persons)
                """)
                orphaned_logs = cursor.rowcount
                self.cleanup_stats['database_records_cleaned'] += orphaned_logs
                logger.info(f"Deleted {orphaned_logs} orphaned recognition log records")
                
                # Clean up old recognition logs (older than 90 days)
                cursor.execute("""
                    DELETE FROM recognition_logs 
                    WHERE recognized_at < datetime('now', '-90 days')
                """)
                old_logs = cursor.rowcount
                self.cleanup_stats['database_records_cleaned'] += old_logs
                logger.info(f"Deleted {old_logs} old recognition log records")
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Database cleanup failed: {e}")
    
    def cleanup_orphaned_files(self):
        """Remove files that are no longer referenced in database"""
        logger.info("=== Cleaning Orphaned Files ===")
        
        db_path = os.path.join(self.base_dir, "data", "face_database.db")
        if not os.path.exists(db_path):
            return
        
        try:
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                
                # Get all referenced file paths
                cursor.execute("SELECT image_path, thumbnail_path, embedding_path FROM face_images")
                referenced_files = set()
                for row in cursor.fetchall():
                    for path in row:
                        if path:
                            referenced_files.add(path)
                
                # Check thumbnails directory
                thumbnails_dir = os.path.join(self.base_dir, "data", "thumbnails")
                if os.path.exists(thumbnails_dir):
                    for file_name in os.listdir(thumbnails_dir):
                        file_path = os.path.join(thumbnails_dir, file_name)
                        if os.path.isfile(file_path) and file_path not in referenced_files:
                            try:
                                size = os.path.getsize(file_path)
                                os.remove(file_path)
                                self.cleanup_stats['files_deleted'] += 1
                                self.cleanup_stats['space_freed_mb'] += size / (1024 * 1024)
                                logger.info(f"Deleted orphaned thumbnail: {file_name}")
                            except Exception as e:
                                logger.warning(f"Failed to delete {file_path}: {e}")
                
        except Exception as e:
            logger.error(f"Orphaned files cleanup failed: {e}")
    
    def cleanup_duplicate_identities(self):
        """Remove duplicate identity files"""
        logger.info("=== Cleaning Duplicate Identities ===")
        
        identities_dir = os.path.join(self.base_dir, "data", "face_identities")
        if not os.path.exists(identities_dir):
            return
        
        # Group files by person name
        identity_files = {}
        for file_name in os.listdir(identities_dir):
            if file_name.endswith('_identity.json'):
                person_name = file_name.replace('_identity.json', '')
                if person_name not in identity_files:
                    identity_files[person_name] = []
                identity_files[person_name].append(file_name)
        
        # Remove duplicates (keep the most recent)
        for person_name, files in identity_files.items():
            if len(files) > 1:
                # Sort by modification time (newest first)
                files_with_time = []
                for file_name in files:
                    file_path = os.path.join(identities_dir, file_name)
                    mtime = os.path.getmtime(file_path)
                    files_with_time.append((file_name, mtime))
                
                files_with_time.sort(key=lambda x: x[1], reverse=True)
                
                # Keep the newest, delete the rest
                for file_name, _ in files_with_time[1:]:
                    file_path = os.path.join(identities_dir, file_name)
                    try:
                        size = os.path.getsize(file_path)
                        os.remove(file_path)
                        self.cleanup_stats['files_deleted'] += 1
                        self.cleanup_stats['space_freed_mb'] += size / (1024 * 1024)
                        logger.info(f"Deleted duplicate identity: {file_name}")
                    except Exception as e:
                        logger.warning(f"Failed to delete {file_path}: {e}")
    
    def cleanup_documentation(self):
        """Remove outdated documentation files"""
        logger.info("=== Cleaning Documentation ===")
        
        outdated_docs = [
            "FACE_ID_IMPROVEMENTS_SUMMARY.md",
            "VIDEO_REGISTRATION_DEBUG_FIXES.md",
            "ULTIMATE_ACCURACY_COMPLETE.md",
            "ULTIMATE_MULTIMODEL_COMPLETE.md",
            "DEEPFACE_INTEGRATION_COMPLETE.md",
            "FACE_RECOGNITION_IMPROVEMENTS_COMPLETE.md",
            "FRAME_CAPTURE_OPTIMIZATION.md"
        ]
        
        for doc_file in outdated_docs:
            doc_path = os.path.join(self.base_dir, doc_file)
            if os.path.exists(doc_path):
                try:
                    size = os.path.getsize(doc_path)
                    os.remove(doc_path)
                    self.cleanup_stats['files_deleted'] += 1
                    self.cleanup_stats['space_freed_mb'] += size / (1024 * 1024)
                    logger.info(f"Deleted outdated doc: {doc_file}")
                except Exception as e:
                    logger.warning(f"Failed to delete {doc_path}: {e}")
    
    def cleanup_unused_scripts(self):
        """Remove unused scripts and files"""
        logger.info("=== Cleaning Unused Scripts ===")
        
        unused_files = [
            "face_id_system.py",  # Old version
            "examples.py",  # Example file
            "minimal_face_id.py",  # Minimal version
            "install_compatible.py",  # Installation scripts
            "install_simple.py",
            "install_windows.py",
            "video_registration_cli.py",  # CLI version
            "external_dataset_trainer.py",  # External training
            "requirements_compatible.txt"  # Old requirements
        ]
        
        for file_name in unused_files:
            file_path = os.path.join(self.base_dir, file_name)
            if os.path.exists(file_path):
                try:
                    size = os.path.getsize(file_path)
                    os.remove(file_path)
                    self.cleanup_stats['files_deleted'] += 1
                    self.cleanup_stats['space_freed_mb'] += size / (1024 * 1024)
                    logger.info(f"Deleted unused file: {file_name}")
                except Exception as e:
                    logger.warning(f"Failed to delete {file_path}: {e}")
    
    def optimize_database(self):
        """Optimize database performance"""
        logger.info("=== Optimizing Database ===")
        
        db_path = os.path.join(self.base_dir, "data", "face_database.db")
        if not os.path.exists(db_path):
            return
        
        try:
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                
                # Vacuum database to reclaim space
                cursor.execute("VACUUM")
                logger.info("Database vacuumed")
                
                # Analyze database for better query performance
                cursor.execute("ANALYZE")
                logger.info("Database analyzed")
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Database optimization failed: {e}")
    
    def run_comprehensive_cleanup(self):
        """Run all cleanup operations"""
        logger.info("=== Starting Comprehensive System Cleanup ===")
        
        start_time = datetime.now()
        
        # Run all cleanup operations
        self.cleanup_test_files()
        self.cleanup_pycache()
        self.cleanup_old_uploads()
        self.cleanup_orphaned_video_frames()
        self.cleanup_database()
        self.cleanup_orphaned_files()
        self.cleanup_duplicate_identities()
        self.cleanup_documentation()
        self.cleanup_unused_scripts()
        self.optimize_database()
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        # Print cleanup summary
        logger.info("=== Cleanup Summary ===")
        logger.info(f"Files deleted: {self.cleanup_stats['files_deleted']}")
        logger.info(f"Directories deleted: {self.cleanup_stats['directories_deleted']}")
        logger.info(f"Database records cleaned: {self.cleanup_stats['database_records_cleaned']}")
        logger.info(f"Space freed: {self.cleanup_stats['space_freed_mb']:.2f} MB")
        logger.info(f"Cleanup duration: {duration.total_seconds():.2f} seconds")
        
        logger.info("=== System Cleanup Complete ===")

def main():
    """Main cleanup function"""
    cleanup = SystemCleanup()
    cleanup.run_comprehensive_cleanup()

if __name__ == "__main__":
    main()
