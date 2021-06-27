from ineuron import config

def launchTensorBoard():
    import os
    os.system('tensorboard --logdir=' + config.OUTPUT_PATH)
    return

import threading
t = threading.Thread(target=launchTensorBoard, args=([]))
t.start()