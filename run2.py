from picamera import PiCamera
from time import sleep
import time
import io
from PIL import Image
#import keyboard
from gpiozero import Robot
import tensorflow as tf
import os
print('import finished')
import numpy as np
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
camera.resolution = (32,32)
camera.framerate = 15
time.sleep(1)

#camera.start_preview()

#outputs = [io.BytesIO() for i in range(60)]
#camera.capture_sequence(outputs,'jpeg',use_video_port=True)
#v1 = tf.Variable(tf.zeros([10]))
#saver = tf.train.Saver()
#sess = tf.Session()
#saver.restore(sess,os.path.abspath(os.path.join(os.getcwd(),'model_3')))

graph = tf.get_default_graph()

LABELS = 5 # Number of different types of labels (1-10)
WIDTH = 32 # width / height of the image
CHANNELS = 3 # Number of colors in the image (greyscale)

VALID = 30 # Validation data size

STEPS = 20000 #20000   # Number of steps to run
BATCH = 100 # Stochastic Gradient Descent batch size
PATCH = 5 # Convolutional Kernel size
DEPTH = 12 #32 # Convolutional Kernel depth size == Number of Convolutional Kernels
HIDDEN = 100 #1024 # Number of hidden neurons in the fully connected layer

LR = 0.001 # Learning rate

tf_data = tf.placeholder(tf.float32, shape=(None, WIDTH, WIDTH, CHANNELS))
tf_labels = tf.placeholder(tf.float32, shape=(None, LABELS))

w1 = tf.Variable(tf.truncated_normal([PATCH, PATCH, CHANNELS, DEPTH], stddev=0.1))
b1 = tf.Variable(tf.zeros([DEPTH]))
w2 = tf.Variable(tf.truncated_normal([PATCH, PATCH, DEPTH, 2*DEPTH], stddev=0.1))
b2 = tf.Variable(tf.constant(1.0, shape=[2*DEPTH]))
w3 = tf.Variable(tf.truncated_normal([WIDTH // 4 * WIDTH // 4 * 2*DEPTH, HIDDEN], stddev=0.1))
b3 = tf.Variable(tf.constant(1.0, shape=[HIDDEN]))
w4 = tf.Variable(tf.truncated_normal([HIDDEN, LABELS], stddev=0.1))
b4 = tf.Variable(tf.constant(1.0, shape=[LABELS]))

saver = tf.train.Saver()
sess = tf.Session()
saver.restore(sess,os.path.abspath(os.path.join(os.getcwd(),'model_3')))


print('model loaded')

def logits(data):
    # Convolutional layer 1
    x = tf.nn.conv2d(data, w1, [1, 1, 1, 1], padding='SAME')
    x = tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
    x = tf.nn.relu(x + b1)
    # Convolutional layer 2
    x = tf.nn.conv2d(x, w2, [1, 1, 1, 1], padding='SAME')
    x = tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
    x = tf.nn.relu(x + b2)
    # Fully connected layer
    x = tf.reshape(x, (-1, WIDTH // 4 * WIDTH // 4 * 2*DEPTH))
    x = tf.nn.relu(tf.matmul(x, w3) + b3)
    return tf.matmul(x, w4) + b4

# Prediction:
tf_pred = tf.nn.softmax(logits(tf_data))


count = 0
sleeptimer = 0.3

camera.start_preview()
def take_noods(s):
    camera.capture('train/' + str(count) + s + '.jpg')
while True:
	pic = np.empty((32,32,3),dtype=np.float32)
	camera.capture(pic,'rgb')
	test_pred = sess.run(tf_pred,feed_dict={tf_data:np.array([pic])})
	test_label = np.argmax(test_pred,axis=1)
	print(test_label[0])
	if (test_label[0] == 0):
		robot.forward (maxSpeed)
		time.sleep(sleeptimer)
		robot.stop()
	elif (test_label[0] == 1):
		robot.forward (maxSpeed, curve_left = turnSpeed + .4)
		time.sleep(sleeptimer)
		robot.stop()
	elif (test_label[0] == 2):
		robot.forward (maxSpeed, curve_right = turnSpeed + .4)
		time.sleep(sleeptimer)
		robot.stop()
	elif (test_label[0] == 3):
		#robot.backward (maxSpeed)
                #time.sleep(sleeptimer)
		robot.stop()
	elif (test_label[0] == 4):
		#robot. (maxSpeed)
		#time.sleep(sleeptimer)
		robot.stop()
		
while False:
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

