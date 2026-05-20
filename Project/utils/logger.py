import sys
import os

class Logger(object):
    def __init__(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.terminal = sys.stdout
        self.log = open(filepath, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        # Flush immediately so long-running labs (e.g. lab08 SVM grid) write progress in real
        # time both to the terminal and to the log file - useful when the run is monitored or
        # interrupted.
        self.terminal.flush()
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def setup_logger(filepath):
    sys.stdout = Logger(filepath)
