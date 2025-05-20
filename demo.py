from src.logger import logging
from src.exception import SRCException 
import sys

logging.info("hellow everyone ")

try:
    a = 2/0
except Exception as e:
    raise SRCException(e, sys)
    