import os
import json
from datetime import datetime
from typing import Optional


def log_ai_function(
    response: str,
    file_name: str = "agent_output.py",
    log: bool = False,
    log_path: Optional[str] = None,
    overwrite: bool = True,
) -> tuple[str, str]:
    """Log AI-generated function to file."""
    if not log:
        return None, None
    
    if log_path is None:
        log_path = os.path.join(os.getcwd(), "logs")
    
    # Create logs directory if it doesn't exist
    os.makedirs(log_path, exist_ok=True)
    
    # Generate unique filename if overwrite is False
    if not overwrite:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name, ext = os.path.splitext(file_name)
        file_name = f"{name}_{timestamp}{ext}"
    
    full_path = os.path.join(log_path, file_name)
    
    # Write response to file
    with open(full_path, 'w', encoding='utf-8') as f:
        f.write(response)
    
    return full_path, file_name


def log_ai_error(
    error_message: str,
    file_name: str = "error.log",
    log: bool = False,
    log_path: Optional[str] = None,
    overwrite: bool = False,
) -> Optional[str]:
    """Log AI error to file."""
    if not log:
        return None
    
    if log_path is None:
        log_path = os.path.join(os.getcwd(), "logs")
    
    os.makedirs(log_path, exist_ok=True)
    
    # Generate unique filename if overwrite is False
    if not overwrite:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name, ext = os.path.splitext(file_name)
        file_name = f"{name}_{timestamp}{ext}"
    
    full_path = os.path.join(log_path, file_name)
    
    # Write error to file
    with open(full_path, 'a', encoding='utf-8') as f:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"[{timestamp}] {error_message}\n")
    
    return full_path


def create_log_directory(log_path: Optional[str] = None) -> str:
    """Create log directory if it doesn't exist."""
    if log_path is None:
        log_path = os.path.join(os.getcwd(), "logs")
    
    os.makedirs(log_path, exist_ok=True)
    return log_path
