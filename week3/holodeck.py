#!/usr/bin/env python3
import numpy as np
import cv2
import pygame as pg

from Holodeck import Holodeck, Agents
from Holodeck.Environments import HolodeckEnvironment
from Holodeck.Sensors import Sensors

def filter_keys(keys, command):
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

    return command

def uav_holodeck():
    # Make holodeck environment
    env = Holodeck.make("ConiferForest")
    env.reset()

    gameDisplay = pg.display.set_mode((512, 512))

    cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)

    # This command tells the UAV to not roll or pitch, but to constantly yaw left at 10m altitude.
    pg.event.pump()
    keys = pg.key.get_pressed()
    command = [0.0, 0.0, 0.0, 0.0]
    command = filter_keys(keys, command)

    while (not keys[pg.K_ESCAPE]):
        state, reward, terminal, _ = env.step(command)

        # To access specific sensor data:
        frame = cv2.cvtColor(state[Sensors.PRIMARY_PLAYER_CAMERA], cv2.COLOR_BGRA2RGB)
        frame = np.rot90(frame)
        frame = pg.surfarray.make_surface(frame)
        gameDisplay.blit(frame, (0,0))
        pg.display.update()

        pg.event.pump()
        keys = pg.key.get_pressed()
        command = filter_keys(keys, command)

if __name__ == "__main__":
    pg.init()
    uav_holodeck()
    cv2.destroyAllWindows()
    pg.quit()
    print("Finished")
