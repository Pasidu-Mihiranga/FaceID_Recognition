"""
Authentication and Rate Limiting Middleware
"""

from functools import wraps
from flask import request, jsonify
import os
import time
import logging

logger = logging.getLogger(__name__)

# In-memory storage for rate limiting: {ip_address: [timestamp1, timestamp2, ...]}
rate_limit_cache = {}

def require_api_key(f):
    """
    Decorator to enforce API Key authentication.
    Checks the 'X-API-Key' header.
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = os.getenv("FACEID_API_KEY")
        if not api_key:
            # If no API key configured in env, bypass check (log warning)
            return f(*args, **kwargs)
            
        request_key = request.headers.get("X-API-Key")
        if not request_key or request_key != api_key:
            logger.warning(f"Unauthorized access attempt to {request.path} from IP {request.remote_addr}")
            return jsonify({"error": "Unauthorized: Invalid or missing API key"}), 401
            
        return f(*args, **kwargs)
    return decorated_function

def rate_limit(limit=10, period=60):
    """
    Decorator for sliding-window rate limiting based on client IP.
    
    Args:
        limit: Max number of requests allowed in the period.
        period: Time window in seconds.
    """
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            ip = request.remote_addr
            now = time.time()
            
            # Clean up old records for this IP
            if ip in rate_limit_cache:
                rate_limit_cache[ip] = [t for t in rate_limit_cache[ip] if now - t < period]
            else:
                rate_limit_cache[ip] = []
                
            # Check rate limit
            if len(rate_limit_cache[ip]) >= limit:
                logger.warning(f"Rate limit exceeded for IP {ip} on {request.path}")
                return jsonify({"error": "Rate limit exceeded. Please try again later."}), 429
                
            # Log current request timestamp
            rate_limit_cache[ip].append(now)
            return f(*args, **kwargs)
        return decorated_function
    return decorator
