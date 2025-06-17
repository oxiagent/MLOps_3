#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script for monitoring system resources in the SageMaker container.
Runs in the background mode and periodically writes resource information to logs.
"""

import os
import sys
import time
import logging
import psutil
import threading
import datetime
import json
import traceback

# Logging configuration
LOG_DIR = "/opt/ml/output/data"
os.makedirs(LOG_DIR, exist_ok=True)

# Log file name format with timestamp
log_timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
LOG_FILE = os.path.join(LOG_DIR, f"monitor-{log_timestamp}.log")

# Logger setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("SageMaker-Monitor")

# Monitoring interval in seconds
MONITOR_INTERVAL = int(os.environ.get("MONITOR_INTERVAL", "60"))

def get_system_info():
    """Get system resource information"""
    try:
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Memory usage
        memory = psutil.virtual_memory()
        
        # Disk usage
        disk = psutil.disk_usage('/')
        
        # Process information
        current_process = psutil.Process(os.getpid())
        current_process_memory = current_process.memory_info().rss / (1024 * 1024)  # in MB
        
        # Create metrics dictionary
        metrics = {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "cpu": {
                "percent": cpu_percent,
                "cores": psutil.cpu_count(logical=True)
            },
            "memory": {
                "total_mb": memory.total / (1024 * 1024),
                "available_mb": memory.available / (1024 * 1024),
                "used_percent": memory.percent
            },
            "disk": {
                "total_gb": disk.total / (1024 * 1024 * 1024),
                "free_gb": disk.free / (1024 * 1024 * 1024),
                "used_percent": disk.percent
            },
            "process": {
                "pid": os.getpid(),
                "memory_mb": current_process_memory,
                "threads": threading.active_count()
            }
        }
        
        return metrics
    except Exception as e:
        logger.error(f"Error getting system info: {str(e)}")
        logger.error(traceback.format_exc())
        return {"error": str(e)}

def monitor_resources():
    """Main monitoring function, runs in a separate thread"""
    logger.info(f"Starting resource monitoring (interval: {MONITOR_INTERVAL}s)")
    
    try:
        while True:
            # Get resource information
            metrics = get_system_info()
            
            # Write to log
            logger.info(f"Resource metrics: {json.dumps(metrics)}")
            
            # Pause before next metrics collection
            time.sleep(MONITOR_INTERVAL)
    except Exception as e:
        logger.error(f"Monitoring thread error: {str(e)}")
        logger.error(traceback.format_exc())

def start_monitoring():
    """Start monitoring in a separate thread"""
    try:
        # Record startup information
        logger.info("=" * 80)
        logger.info("Starting SageMaker container resource monitor")
        logger.info(f"Monitor interval: {MONITOR_INTERVAL} seconds")
        logger.info(f"Log file: {LOG_FILE}")
        logger.info("=" * 80)
        
        # Create and start monitoring thread
        monitor_thread = threading.Thread(target=monitor_resources, daemon=True)
        monitor_thread.start()
        
        logger.info("Monitoring thread started")
        return monitor_thread
    except Exception as e:
        logger.error(f"Failed to start monitoring: {str(e)}")
        logger.error(traceback.format_exc())
        return None

if __name__ == "__main__":
    # Start monitoring when script is run
    monitor_thread = start_monitoring()
    
    # If this is the main process, keep it running
    if monitor_thread and not os.environ.get("MONITOR_BACKGROUND", False):
        try:
            while monitor_thread.is_alive():
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Monitoring stopped by user")
            sys.exit(0) 