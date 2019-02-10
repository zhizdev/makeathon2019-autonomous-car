from picamera import PiCamera
from time import sleep
import time
import io
from PIL import Image

camera = PiCamera()
camera.resolution = (640,360)
camera.framerate = 15
time.sleep(1)

start = time.time()

camera.start_preview()

outputs = [io.BytesIO() for i in range(60)]
camera.capture_sequence(outputs,'jpeg',use_video_port=True)

count = 0
for frame in outputs:
    rawIO = frame
    rawIO.seek(0)
    byteImg = Image.open(rawIO)
    count += 1
    filename = "image" + str(count) + ".jpeg"
    byteImg.save(filename,'JPEG')

#camera.start_preview()
#sleep(10)
camera.stop_preview()

end = time.time()
print((end-start)/60)
