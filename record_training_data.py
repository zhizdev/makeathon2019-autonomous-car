from picamera import PiCamera
from time import sleep
import time
import io
from PIL import Image
import keyboard
from gpiozero import Robot

""" DON'T FKING CHANGE THIS!!! """
leftPinF = 26
leftPinB = 19
rightPinF = 13
rightPinB = 6
""" DON'T FKING CHANGE THIS!!! """

maxSpeed = 0.4 
turnSpeed = 0.2 

robot = Robot (left = (leftPinF, leftPinB), right = (rightPinF, rightPinB))

camera = PiCamera()
camera.resolution = (640,360)
camera.framerate = 15
time.sleep(1)

#camera.start_preview()

#outputs = [io.BytesIO() for i in range(60)]
#camera.capture_sequence(outputs,'jpeg',use_video_port=True)

count = 0
sleeptimer = 0.3

camera.start_preview()
def take_noods(s):
    camera.capture('train/' + str(count) + s + '.jpg')

while True:
        #dir = sys.stdin.read (1)
		if keyboard.is_pressed ("w"):
			take_noods('w')
			robot.forward (maxSpeed)
			time.sleep(sleeptimer)
			robot.stop()
		elif keyboard.is_pressed ("a"):
			take_noods('a')
			#robot.right(.2)
			#robot.forward (maxSpeed, curve_left = turnSpeed + .2)
			robot.forward (maxSpeed, curve_right = turnSpeed + .4)
			time.sleep(sleeptimer)
			robot.stop()
		elif keyboard.is_pressed ("s"):
			take_noods('s')
			robot.backward (maxSpeed)
			time.sleep(sleeptimer)
			robot.stop()
		elif keyboard.is_pressed ("d"):
			take_noods('d')
			#robot.left(.2)
			robot.forward (maxSpeed, curve_left = turnSpeed + .4)	
			time.sleep(sleeptimer)
			robot.stop()
			#robot.forward (maxSpeed, curve_right = turnSpeed + .2)
            
		elif keyboard.is_pressed ("o"):
			break
            
		elif keyboard.is_pressed ("x"):
			take_noods('x')
			robot.stop()
			
		else:
			robot.stop ()
			continue
		count += 1

camera.stop_preview()

print('training saved')
