"""
Structured Logging Configuration for Face ID System
Supports standard text formatting and structured JSON logging.
"""

import logging
import json
import os
import sys

class JSONFormatter(logging.Formatter):
    """JSON log formatter"""
    def format(self, record):
        log_record = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage()
        }
        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_record)

def setup_logging(level=logging.INFO):
    """Set up structured JSON logging if ENABLE_JSON_LOGGING=true"""
    use_json = os.getenv("ENABLE_JSON_LOGGING", "false").lower() == "true"
    
    root_logger = logging.getLogger()
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        
    handler = logging.StreamHandler(sys.stdout)
    if use_json:
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter('[%(asctime)s] %(levelname)s in %(name)s: %(message)s')
        
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)
    root_logger.setLevel(level)
