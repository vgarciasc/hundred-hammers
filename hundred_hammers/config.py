import os
os.environ['PYTHONWARNINGS']='ignore'

import logging
logging.basicConfig(format='[%(levelname)s] %(asctime)s: %(message)s')
automl_logger = logging.getLogger(__name__)
automl_logger.setLevel(logging.INFO)
