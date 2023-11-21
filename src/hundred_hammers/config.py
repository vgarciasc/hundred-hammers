"""
Global configuration for the library.
"""

import os
import logging

os.environ["PYTHONWARNINGS"] = "ignore"

logging.basicConfig(format="[%(levelname)s] %(asctime)s: %(message)s")
hh_logger = logging.getLogger(__name__)
hh_logger.setLevel(logging.INFO)
