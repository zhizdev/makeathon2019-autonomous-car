# makeathon2019-autonomous-car
Raspberry Pi 3 Powered Autonomous Car with Vision.

## Inspiration
Inspired by recent progress in autonomous cars, we set out to create our own small scale version of an autonomous car with a Raspberry Pi 3. We connected a camera to the Pi and deployed a trained machine learning model to direct the car's motion: forward, turn left, turn right, and stop. 

## What it does
The Pi camera takes an image, lowers the resolution to 32 pixels by 32 pixels, and then feeds it into a convolutional neural network pre-trained on a laptop. After receiving a classification, the car moves in the classified direction before stopping to take another image. The goal of the system is to make the car move within the track and not run off. 

## How I built it
Building the actual car was challenging as we did not have a kit that could fit all the motor controls and the camera mount. We put together a Frankenstein car and 3D printed a camera mount. We then began collecting data by manually driving the car on the track while recording picture action pairs. For every step we tell the car to move forward, a picture is captured and labeled as forward. The same follows for turning and stopping. 

With the labeled images, we constructed a small convolutional neural network with 2 convolutional layers. We scaled all the images to 32 pixels by 32 pixels before training. We were only able to achieve a 62% validation accuracy on the data set. 

## Challenges I ran into
The model was unable to guide the vehicle through the entirety of the track. We collected more data by extracting frames from a video feed on the car, but we believe it may have introduced more noisy data and caused our model to perform poorly.

## Accomplishments that I'm proud of
While the model can be refined with more data, we successfully built the entire system for a vision based autonomous car. We are able to feed the model more images and make our car drive more precisely. 

## What I learned
The most important part about any machine learning system is a good and large set of training labels. We were not able to gather enough clean labels within the constraints of 36 hours, but we believe with more data, our model performance will improve dramatically. 

## What's next for makeathon2019-autonomous-car
We will test our car on a different track with more custom training data. 
