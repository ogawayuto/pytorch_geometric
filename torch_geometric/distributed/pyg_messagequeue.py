import torch
import messagequeue

# The message queue python interface for PyG
class PyGMessageQueue():
    def __init__(self, shm_size=134217728): # 1 GB for the shared memory size
        self.queue = messagequeue.shm(shm_size)
    
    def put(self, input):
        if isinstance(input, dict):
            self.queue.WriteDictOfTensors(input)
        else:
            print("Error - input is not a dictionary")
        return
    
    def get(self) -> dict:
        return self.queue.ReadDictOfTensors()

    # for monitoring the internal operations of the message queue
    def monitor(self):
        return self.queue.Monitor()



    
