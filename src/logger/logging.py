import logging
import os


logger=logging.getLogger('app.py')
logger.setLevel(logging.INFO)
log_path = os.path.join("logs", "app.log")  # logs/app.log
os.makedirs("logs", exist_ok=True)          # Create folder if it doesn't exist
handler = logging.FileHandler(log_path, mode='a')
fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
datefmt = "%Y-%m-%d %H:%M:%S"
handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
logger.addHandler(handler)
