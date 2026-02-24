import time
from positiondetection import PositionDetection

posdetect = PositionDetection(0, 7)
ptime = 0
ctime = 0

while True:
    print(posdetect.detect_position())

    ctime = time.time()
    fps = 1 / (ctime - ptime)
    ptime = ctime
    print(f'Fps: {fps}', end='  ')
    time.sleep(0.002)
