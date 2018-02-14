# This is for problems with clashing opencv versions from ROS installations
import sys
rm_python2 = []
for p in sys.path:
    if p.find('python2') != -1:
        rm_python2.append(p)
for p in rm_python2:
    sys.path.remove(p)

import cv2
import numpy as np
import math
import time

from uav_sim import UAVSim

urban_world = 'UrbanCity'
forest_world = 'EuropeanForest'
redwood_world = 'RedwoodForest'

def holodeck_sim():
    uav_sim = UAVSim(urban_world)
    uav_sim.init_plots(plotting_freq=5) # Commenting this line would disable plotting

    while True:
        # This is the main loop where the simulation is updated
        uav_sim.step_sim()

if __name__ == "__main__":
    holodeck_sim()
    print("Finished")
