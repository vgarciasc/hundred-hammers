import os
os.environ['PYTHONWARNINGS']='ignore'

import logging
logging.basicConfig(format='[%(levelname)s] %(asctime)s: %(message)s')
hh_logger = logging.getLogger(__name__)
hh_logger.setLevel(logging.INFO)
