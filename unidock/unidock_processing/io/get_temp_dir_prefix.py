import os
import socket
from datetime import datetime

def get_temp_dir_prefix(command_name='docking'):
    hostname = socket.gethostname()
    pid = os.getpid()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    temp_dir_prefix = f'{command_name}_{hostname}_{pid}_{timestamp}_'

    return temp_dir_prefix
