from gpiozero import Robot
import keyboard

""" DON'T FKING CHANGE THIS!!! """
leftPinF = 19
leftPinB = 26
rightPinF = 6
rightPinB = 23
""" DON'T FKING CHANGE THIS!!! """

maxSpeed = 0.5 
turnSpeed = 0.2 

robot = Robot (left = (leftPinF, leftPinB), right = (rightPinF, rightPinB))

def main ():
    while True:
        #dir = sys.stdin.read (1)
        if keyboard.is_pressed ("w"):
            robot.forward (maxSpeed)
        elif keyboard.is_pressed ("a"):
            robot.forward (maxSpeed, curve_left = turnSpeed + .2)
        elif keyboard.is_pressed ("d"):
            robot.forward (maxSpeed, curve_right = turnSpeed)
        elif keyboard.is_pressed ("s"):
            robot.backward (maxSpeed)
        else:
            robot.stop ()
    
    """ Have the same thing for backwards. """

if __name__ == "__main__":
    """ This is executed when run from the command line """
    main()
