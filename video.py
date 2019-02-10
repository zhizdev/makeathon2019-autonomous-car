import picamera
import time
with picamera.PiCamera() as camera:
    camera.resolution = (64, 64)
    camera.framerate = 15
    camera.start_recording('train_vid/stop_'+ str(time.ctime()) + '.h264')
    camera.wait_recording(3)
    camera.stop_recording()
