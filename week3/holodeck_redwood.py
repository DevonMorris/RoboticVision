#!/usr/bin/env python3
import numpy as np
import cv2
import pygame as pg
import matplotlib.pyplot as plt
import transforms3d

from enum import Enum

from Holodeck import Holodeck, Agents
from Holodeck.Environments import HolodeckEnvironment
from Holodeck.Sensors import Sensors

class Filter(Enum):
    NO_FILTER = 0
    CANNY = 1
    SOBEL = 2
    SOBELX = 3
    SOBELY = 4
    GAUSSIAN = 5
    GRADIENT = 6

def filter_keys(keys, command, filt):
    if keys[pg.K_a]:
        command[2] += -0.03
    if keys[pg.K_d]:
        command[2] += 0.03
    if keys[pg.K_w]:
        command[3] += 0.10
    if keys[pg.K_s]:
        command[3] -= 0.10
    if keys[pg.K_UP]:
        command[1] -= 0.01
    if keys[pg.K_DOWN]:
        command[1] += 0.01
    if keys[pg.K_LEFT]:
        command[0] -= 0.01
    if keys[pg.K_RIGHT]:
        command[0] += 0.01
    if keys[pg.K_0]:
        filt = Filter.NO_FILTER
    if keys[pg.K_1]:
        filt = Filter.CANNY
    if keys[pg.K_2]:
        filt = Filter.SOBEL
    if keys[pg.K_3]:
        filt = Filter.SOBELX
    if keys[pg.K_4]:
        filt = Filter.SOBELY
    if keys[pg.K_5]:
        filt = Filter.GAUSSIAN
    if keys[pg.K_6]:
        filt = Filter.GRADIENT

    return command, filt

def filter_image(frame, filt):
    if filt == Filter.NO_FILTER:
        pass
    elif filt == Filter.CANNY:
        frame = cv2.Canny(frame, 100, 300)
    elif filt == Filter.SOBEL:
        frame = cv2.Sobel(frame, cv2.CV_32F, 1, 1)
    elif filt == Filter.SOBELX:
        frame = cv2.Sobel(frame, cv2.CV_32F, 1, 0)
    elif filt == Filter.SOBELY:
        frame = cv2.Sobel(frame, cv2.CV_32F, 0, 1)
    elif filt == Filter.GAUSSIAN:
        frame = cv2.GaussianBlur(frame, (9,9), 2.0)
    elif filt == Filter.GRADIENT:
        frame = cv2.Laplacian(frame, cv2.CV_32F)
    return frame

def uav_holodeck():
    # Make holodeck environment
    env = Holodeck.make("RedwoodForest")
    env.reset()

    gameDisplay = pg.display.set_mode((512, 512))

    cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)

    # This command tells the UAV to not roll or pitch, but to constantly yaw left at 10m altitude.
    pg.event.pump()
    keys = pg.key.get_pressed()
    command = [0.0, 0.0, 0.0, 0.0]
    filt = 0
    location = []
    orientation = []
    velocity = []
    imu = []


    command, filt = filter_keys(keys, command, filt)

    while (not keys[pg.K_ESCAPE]):
        state, reward, terminal, _ = env.step(command)
        location.append(np.copy(state[Sensors.LOCATION_SENSOR]))
        orientation.append(transforms3d.euler.mat2euler(state[Sensors.ORIENTATION_SENSOR], "rxyz"))
        velocity.append(np.copy(state[Sensors.VELOCITY_SENSOR]))
        imu.append(np.copy(state[Sensors.IMU_SENSOR]))

        # To access specific sensor data:
        frame = cv2.cvtColor(state[Sensors.PRIMARY_PLAYER_CAMERA], cv2.COLOR_BGRA2RGB)
        frame = np.rot90(frame)

        frame = filter_image(frame, filt)

        frame = pg.surfarray.make_surface(frame)
        gameDisplay.blit(frame, (0,0))
        pg.display.update()

        pg.event.pump()
        keys = pg.key.get_pressed()
        command, filt = filter_keys(keys, command, filt)

    cv2.destroyAllWindows()
    pg.quit()

    location = np.array(location).reshape((-1,3))
    orientation = np.array(orientation).reshape((-1,3))
    velocity = np.array(velocity).reshape((-1,3))
    imu = np.array(imu).reshape((-1,6))
    
    print(location.copy())

    plt.figure(0)
    plt.plot(location)
    plt.title("Location")
    plt.legend(("$n$", "$e$", "$h$"))
    plt.show()

    plt.figure(1)
    plt.plot(orientation)
    plt.title("Orientation")
    plt.legend(("$\phi$", r"$\theta$", "$\psi$"))
    plt.show()

    plt.figure(2)
    plt.plot(velocity)
    plt.title("Velocity")
    plt.legend(("$u$", "$v$", "$w$"))
    plt.show()

    plt.figure(3)
    plt.plot(imu)
    plt.title("Imu")
    plt.legend(("$a_x$", "$a_y$", "$a_z$", 
        "$g_x$", "$g_y$", "$g_z$"))
    plt.show()
if __name__ == "__main__":
    pg.init()
    uav_holodeck()
    print("Finished")
