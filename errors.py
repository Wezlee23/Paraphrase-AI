"""
Error Handling Module for Flask Paraphraser Application
========================================================
Provides custom exceptions, error handlers, and logging configuration.
"""

import logging
from logging.handlers import TimedRotatingFileHandler
import os
from functools import wraps
from flask import jsonify, render_template, request

# ============================================
# LOGGING CONFIGURATION
# ============================================

def setup_logging(app, log_dir='logs', log_level=logging.INFO):
    """Configure application logging with daily file rotation.
    
    Creates a new log file for each day in the format: app_YYYY-MM-DD.log
    Keeps logs for 30 days by default.
    """
    # Create logs directory if needed
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_file = os.path.join(log_dir, 'app.log')
    
    # Daily rotating file handler (rotates at midnight, keeps 30 days)
    file_handler = TimedRotatingFileHandler(
        log_file,
        when='midnight',
        interval=1,
        backupCount=30,
        encoding='utf-8'
    )
    # Rename log files to include date: app.log -> app_2024-12-24.log
    file_handler.suffix = '_%Y-%m-%d.log'
    file_handler.namer = lambda name: name.replace('.log_', '_').replace('.log', '')
    
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    ))
    file_handler.setLevel(log_level)
    
    # Console handler for development
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(
        '%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S'
    ))
    console_handler.setLevel(log_level)
    
    # Configure app logger
    app.logger.addHandler(file_handler)
    app.logger.addHandler(console_handler)
    app.logger.setLevel(log_level)
    
    # Also configure root logger for third-party libraries
    logging.getLogger().addHandler(file_handler)
    
    app.logger.info('Logging initialized (daily rotation enabled)')


# ============================================
# CUSTOM EXCEPTIONS
# ============================================

class AppError(Exception):
    """Base exception for application errors."""
    def __init__(self, message, status_code=500, payload=None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.payload = payload

    def to_dict(self):
        rv = dict(self.payload or ())
        rv['error'] = self.message
        rv['success'] = False
        return rv


class APIError(AppError):
    """External API call failure."""
    def __init__(self, message="External API error", status_code=502, payload=None):
        super().__init__(message, status_code, payload)


class FileProcessingError(AppError):
    """File upload/processing failure."""
    def __init__(self, message="File processing error", status_code=400, payload=None):
        super().__init__(message, status_code, payload)


class TranscriptionError(AppError):
    """Audio/video transcription failure."""
    def __init__(self, message="Transcription failed", status_code=400, payload=None):
        super().__init__(message, status_code, payload)


class ValidationError(AppError):
    """Input validation failure."""
    def __init__(self, message="Validation error", status_code=400, payload=None):
        super().__init__(message, status_code, payload)


# ============================================
# ERROR RESPONSE HELPERS
# ============================================

def error_response(message, status_code=400, **kwargs):
    """Create a standardized JSON error response."""
    response = {
        'success': False,
        'error': message,
        **kwargs
    }
    return jsonify(response), status_code


def success_response(data=None, message=None, **kwargs):
    """Create a standardized JSON success response."""
    response = {
        'success': True,
        **kwargs
    }
    if data is not None:
        response['data'] = data
    if message:
        response['message'] = message
    return jsonify(response)


# ============================================
# FLASK ERROR HANDLERS REGISTRATION
# ============================================

def register_error_handlers(app):
    """Register all error handlers with the Flask app."""
    
    @app.errorhandler(AppError)
    def handle_app_error(error):
        """Handle custom application errors."""
        app.logger.error(f'AppError: {error.message}')
        if request.is_json or request.headers.get('Accept') == 'application/json':
            return jsonify(error.to_dict()), error.status_code
        return render_template('error.html', 
                               error_code=error.status_code, 
                               error_message=error.message), error.status_code
    
    @app.errorhandler(400)
    def bad_request(error):
        """Handle bad request errors."""
        app.logger.warning(f'Bad Request: {request.url}')
        if request.is_json or request.headers.get('Accept') == 'application/json':
            return error_response('Bad request', 400)
        return render_template('error.html', 
                               error_code=400, 
                               error_message='Bad Request'), 400
    
    @app.errorhandler(404)
    def not_found(error):
        """Handle 404 Not Found errors."""
        app.logger.warning(f'404 Not Found: {request.url}')
        if request.is_json or request.headers.get('Accept') == 'application/json':
            return error_response('Resource not found', 404)
        return render_template('404.html'), 404
    
    @app.errorhandler(413)
    def request_entity_too_large(error):
        """Handle file too large errors."""
        app.logger.warning(f'File too large: {request.url}')
        if request.is_json or request.headers.get('Accept') == 'application/json':
            return error_response('File too large. Please upload a smaller file.', 413)
        return render_template('error.html', 
                               error_code=413, 
                               error_message='File too large'), 413
    
    @app.errorhandler(500)
    def internal_error(error):
        """Handle internal server errors."""
        app.logger.error(f'500 Error: {str(error)}', exc_info=True)
        if request.is_json or request.headers.get('Accept') == 'application/json':
            return error_response('An internal error occurred. Please try again.', 500)
        return render_template('500.html'), 500
    
    @app.errorhandler(Exception)
    def handle_unexpected_error(error):
        """Catch-all for unexpected errors."""
        app.logger.error(f'Unexpected error: {str(error)}', exc_info=True)
        if request.is_json or request.headers.get('Accept') == 'application/json':
            return error_response('An unexpected error occurred', 500)
        return render_template('500.html'), 500


# ============================================
# REQUEST LOGGING MIDDLEWARE
# ============================================

def setup_request_logging(app):
    """Add request/response logging."""
    
    @app.before_request
    def log_request():
        """Log incoming requests."""
        app.logger.debug(f'Request: {request.method} {request.path}')
    
    @app.after_request
    def log_response(response):
        """Log response status."""
        app.logger.debug(f'Response: {response.status_code} for {request.path}')
        return response
