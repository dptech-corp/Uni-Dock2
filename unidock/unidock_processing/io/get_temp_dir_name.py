import os
import socket
from datetime import datetime
import uuid

def get_temp_dir_name(command_name='docking'):
    hostname = socket.gethostname()
    pid = os.getpid()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    rand_suffix = uuid.uuid4().hex[:8]
    temp_dir_name = f'{command_name}_{hostname}_{pid}_{timestamp}_{rand_suffix}'

    return temp_dir_name
