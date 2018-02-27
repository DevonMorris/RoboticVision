import numpy as np
import cv2
import math
import pygame as pg
import transforms3d
from enum import Enum
from pygame.locals import *

from Holodeck import Holodeck, Agents
from Holodeck.Environments import HolodeckEnvironment
from Holodeck.Sensors import Sensors
from state_plotter import Plotter
from PID import PID
import sys
from optic_flow_control import OpticFlowControl

### Command key mappings ###
# Basic commands
ROLL_RIGHT  = K_RIGHT
ROLL_LEFT   = K_LEFT
PITCH_UP    = K_UP
PITCH_DOWN  = K_DOWN
YAW_LEFT    = K_a
YAW_RIGHT   = K_d
ALT_UP      = K_w
ALT_DOWN    = K_s
SPEED_UP    = K_e
SPEED_DOWN  = K_q
# Velocity commands
VEL_FORWARD = K_UP
VEL_BACKWARD= K_DOWN
VEL_RIGHT   = K_RIGHT
VEL_LEFT    = K_LEFT
# System commands
QUIT        = K_ESCAPE
RESET       = K_HOME
PAUSE       = K_SPACE
MANUAL_TOGGLE = K_LCTRL
# Mode control
M_OPTICAL_FLOW = K_o
M_VELOCITY = K_v
M_RAW = K_r

class Mode(Enum):
    RAW = 1
    VELOCITY = 2
    OPTICAL_FLOW = 3

class UAVSim():
    def __init__(self, world):
        ### Parameters
        # Default command
        self.roll_c = 0.0
        self.pitch_c = 0.0
        self.yawrate_c = 0.0
        self.alt_c = 0.0
        self.yaw_c = 0.0
        self.command = np.array([self.roll_c, self.pitch_c, self.yawrate_c, self.alt_c])

        # Rate parameters
        self.roll_min = -math.radians(45)
        self.roll_max = math.radians(45)
        self.pitch_min = -math.radians(45)
        self.pitch_max = math.radians(45)
        self.yawrate_min = -math.radians(360)
        self.yawrate_max = math.radians(360)
        self.altrate_min = 0.1
        self.altrate_max = 0.5
        self.speed_min = 0.0
        self.speed_max = 1.0
        self.speed_rate = 0.05
        self.speed_val = 0

        # Velocity parmeters
        self.vx_c   = 0.0
        self.vy_c   = 0.0
        self.vy_min = -250.0
        self.vy_max = 250.0
        self.vx_min = -250.0
        self.vx_max = 250.0

        # PID controllers
        vx_kp = -self.pitch_max/self.vx_max
        vx_kd = -0.001
        vx_ki = -0.001
        self.vx_pid = PID(vx_kp, vx_kd, vx_ki, u_min=-self.pitch_max, u_max=self.pitch_max)
        vy_kp = 0.5*self.roll_max/self.vy_max
        vy_kd = 0.001
        vy_ki = 0.001
        self.vy_pid = PID(vy_kp, vy_kd, vy_ki, u_min=-self.roll_max, u_max=self.roll_max)

        # Mode
        self.mode = Mode.RAW

        # Teleop
        pg.init()
        self.using_teleop = True
        self.teleop_font = pg.font.Font(None, 50)
        self.teleop_text = "Click here to use teleop"

        # Simulation return variables
        self.sim_state = 0
        self.sim_reward = 0
        self.sim_terminal = 0
        self.sim_info = 0
        self.sim_step = 0
        self.dt = 1.0/30 # 30 Hz

        # Sensor data
        self.camera_sensor = np.zeros((512,512,4))
        self.position_sensor = np.zeros((3,1))
        self.orientation_sensor = np.identity(3)
        self.imu_sensor = np.zeros((6,1))
        self.velocity_sensor = np.zeros((3,1))

        # Default system variables
        self.plotting_states    = False
        self.paused             = False

        # Initialize world
        print("Initializing {0} world".format(world))
        self.env = Holodeck.make(world)
        self.pressed = {PAUSE: False, MANUAL_TOGGLE: False}

        # optic flow
        self.of_control = OpticFlowControl(512,512)

        # prime the environment
        self.reset_sim()
        self.init_camera()


    ######## Plotting Functions ########
    def init_plots(self, plotting_freq):
        self.plotting_states = True
        self.plotter = Plotter(plotting_freq)
        # Define plot names
        plots = ['x',                   'y',                    ['z', 'z_c'],
                 ['xdot', 'xdot_c'],    ['ydot', 'ydot_c'],     'zdot',
                 ['phi', 'phi_c'],      ['theta', 'theta_c'],   ['psi', 'psi_c'],
                 'p',                   'q',                    ['r', 'r_c'],
                 'ax',                  'ay',                   'az'
                 ]
        # Add plots to the window
        for p in plots:
            self.plotter.add_plot(p)

        # Define state vectors for simpler input
        self.plotter.define_state_vector("position", ['x', 'y', 'z'])
        self.plotter.define_state_vector("velocity", ['xdot', 'ydot', 'zdot'])
        self.plotter.define_state_vector("orientation", ['phi', 'theta', 'psi'])
        self.plotter.define_state_vector("imu", ['ax', 'ay', 'az', 'p', 'q', 'r'])
        self.plotter.define_state_vector("command", ['phi_c', 'theta_c', 'r_c', 'z_c'])
        self.plotter.define_state_vector("vel_command", ['xdot_c', 'ydot_c', 'psi_c'])


    ######## Teleop Functions ########
    def init_camera(self):
        SURFACE_WIDTH = 512
        SURFACE_HEIGHT = 512
        self.camera_window = pg.display.set_mode( (SURFACE_WIDTH,SURFACE_HEIGHT) )
        return self.camera_window

    def update_camera_window(self):
        img = self.of_control.annotate(self.camera_sensor)
        img = cv2.cvtColor(img , cv2.COLOR_BGRA2RGB)
        img = np.rot90(np.fliplr(img))
        img = pg.surfarray.make_surface(img)
        self.camera_window.blit(img, (0,0))
        block = self.teleop_font.render(self.teleop_text, True, (255,255,255))
        rect = block.get_rect()
        rect.center = (256, 475)
        self.camera_window.blit(block, rect)
        pg.display.update()

    def process_teleop(self):
        pg.event.pump()
        keys=pg.key.get_pressed()

        self.teleop_system_events(keys)
        self.teleop_mode_control(keys)
        if self.mode == Mode.RAW or self.mode == Mode.VELOCITY:
            if not self.paused:
                self.teleop_commands(keys)


    def teleop_system_events(self,keys):
        # Update event values
        if keys[QUIT]:
            self.exit_sim()

        if keys[RESET]:
            self.teleop_text = "Position reset"
            self.reset_sim()

        if keys[PAUSE]:
            # Only trigger on edge
            if not self.pressed[PAUSE]:
                self.pressed[PAUSE] = True
                # Bitwise xor
                self.paused = self.paused ^ True
                if self.paused == True:
                    self.teleop_text = "Simulation paused"
                else:
                    self.teleop_text = "Simulation resumed"
        elif self.pressed[PAUSE]: # Reset edge variable
            self.pressed[PAUSE] = False
        if self.paused:
            return

    def teleop_mode_control(self, keys):
        if keys[M_RAW]:
            self.mode = Mode.RAW
            self.teleop_text = "Raw Heading Mode"
            self.reset_commands()
        if keys[M_VELOCITY]:
            self.mode = Mode.VELOCITY
            self.teleop_text = "Velocity Command Mode"
            self.reset_commands()
        if keys[M_OPTICAL_FLOW]:
            self.mode = Mode.OPTICAL_FLOW
            self.teleop_text = "Optical Flow Mode"
            self.reset_commands()

    def teleop_commands(self, keys):

        # Lateral motion
        if self.mode == Mode.VELOCITY:
            if keys[VEL_RIGHT]:
                self.vy_c += (self.vy_max - self.vy_min)*.01
                self.vy_c = min(self.vy_c, self.vy_max)
                self.teleop_text = "VEL_RIGHT at {0:.2f}".format(self.vy_c)
            if keys[VEL_LEFT]:
                self.vy_c -= (self.vy_max - self.vy_min)*.01
                self.vy_c = max(self.vy_c, self.vy_min)
                self.teleop_text = "VEL_LEFT at {0:.2f}".format(self.vy_c)
            if keys[VEL_FORWARD]:
                self.vx_c += (self.vx_max - self.vx_min)*.01
                self.vx_c = min(self.vx_c, self.vx_max)
                self.teleop_text = "VEL_FORWARD at {0:.2f}".format(self.vx_c)
            if keys[VEL_BACKWARD]:
                self.vx_c -= (self.vx_max - self.vx_min)*.01
                self.vx_c = max(self.vx_c, self.vx_min)
                self.teleop_text = "VEL_BACKWARD at {0:.2f}".format(self.vx_c)
            # z-rotation
            if keys[YAW_LEFT]:
                self.yawrate_c += (self.yawrate_max - self.yawrate_min)*.01 
                self.yawrate_c = min(self.yawrate_c, self.yawrate_max)
                self.teleop_text = "YAW_LEFT at {0:.2f}".format(self.yawrate_c)
            if keys[YAW_RIGHT]:
                self.yawrate_c -= (self.yawrate_max - self.yawrate_min)*.01 
                self.yawrate_c = max(self.yawrate_c, self.yawrate_min)
                self.teleop_text = "YAW_RIGHT at {0:.2f}".format(self.yawrate_c)

        elif self.mode == Mode.RAW:
            if keys[ROLL_RIGHT]:
                self.roll_c += (self.roll_max - self.roll_min)*.01
                self.roll_c = min(self.roll_max, self.roll_c)
                self.teleop_text = "ROLL_RIGHT to {0:.2f}".format(self.roll_c)
            if keys[ROLL_LEFT]:
                self.roll_c -= (self.roll_max - self.roll_min)*.01
                self.roll_c = max(self.roll_min, self.roll_c)
                self.teleop_text = "ROLL_LEFT to {0:.2f}".format(self.roll_c)
            if keys[PITCH_UP]:
                self.pitch_c += (self.pitch_max - self.pitch_min)*.01                
                self.pitch_c = min(self.pitch_max, self.pitch_c)
                self.teleop_text = "PITCH_UP to {0:.2f}".format(self.pitch_c)
            if keys[PITCH_DOWN]:
                self.pitch_c -= (self.pitch_max - self.pitch_min)*.01                
                self.pitch_c = max(self.pitch_min, self.pitch_c)
                self.teleop_text = "PITCH_DOWN to {0:.2f}".format(self.pitch_c)
            if keys[YAW_LEFT]:
                self.yawrate_c += (self.yawrate_max - self.yawrate_min)*.01 
                self.yawrate_c = min(self.yawrate_c, self.yawrate_max)
                self.teleop_text = "YAW_LEFT at {0:.2f}".format(self.yawrate_c)
            if keys[YAW_RIGHT]:
                self.yawrate_c -= (self.yawrate_max - self.yawrate_min)*.01 
                self.yawrate_c = max(self.yawrate_c, self.yawrate_min)
                self.teleop_text = "YAW_RIGHT at {0:.2f}".format(self.yawrate_c)

        # Altitude
        if keys[ALT_UP]:
            self.alt_c += (self.altrate_min + (self.altrate_max - self.altrate_min)*self.speed_val)
            self.teleop_text = "Altitude raised to {0:.1f}".format(self.alt_c)
        if keys[ALT_DOWN]:
            self.alt_c -= max(((self.altrate_min + (self.altrate_max - self.altrate_min)*self.speed_val), 0))
            self.teleop_text = "Altitude lowered to {0:.1f}".format(self.alt_c)
        # Speed/scaling
        if keys[SPEED_UP]:
            self.speed_val += self.speed_rate
            self.speed_val = min(self.speed_val, self.speed_max)
            self.teleop_text = "Speed raised to {0:.2f}".format(self.speed_val)
        if keys[SPEED_DOWN]:
            self.speed_val -= self.speed_rate
            self.speed_val = max(self.speed_val, self.speed_min)
            self.teleop_text = "Speed lowered to {0:.2f}".format(self.speed_val)


    ######## Control ########
    def compute_control(self):
        if self.mode == Mode.RAW:
            self.compute_raw_control()
        if self.mode == Mode.VELOCITY:
            self.compute_velocity_control()
        if self.mode == Mode.OPTICAL_FLOW:
            self.compute_optical_control()

    def compute_raw_control(self):
        self.set_command(self.roll_c, self.pitch_c, self.yawrate_c, self.alt_c)

    def compute_velocity_control(self):
        # Get current state
        vel = self.get_body_velocity()
        vx = vel[0]
        vy = vel[1]
        yaw = self.get_euler()[2]

        # Compute PID control
        self.pitch_c = self.vx_pid.compute_control(vx, self.vx_c, self.dt)
        self.roll_c = self.vy_pid.compute_control(vy, self.vy_c, self.dt)
        self.set_command(self.roll_c, self.pitch_c, self.yawrate_c, self.alt_c)

    def compute_optical_control(self):
        # Get current state
        euler = self.get_euler()
        phi = euler[0]
        theta = euler[1]

        imu = self.get_imu()

        vel = self.get_body_velocity()
        vx = vel[0]
        vy = vel[1]

        command = self.of_control.control_optic_flow(phi, theta, imu)
        self.roll_c = self.vy_pid.compute_control(vy , command[1], self.dt)
        self.pitch_c = self.vx_pid.compute_control(vx, command[0], self.dt)
        self.yawrate_c = command[2]
        self.alt_c = command[3]

        self.set_command(self.roll_c, self.pitch_c, self.yawrate_c, self.alt_c)


    ######## Data access ########
    def set_command(self, roll, pitch, yawrate, alt):
        self.command = np.array([-roll, pitch, yawrate, alt]) # Roll command is backward in sim

    def extract_sensor_data(self):
        self.camera_sensor      = self.sim_state[Sensors.PRIMARY_PLAYER_CAMERA]
        self.position_sensor    = np.ravel(self.sim_state[Sensors.LOCATION_SENSOR])
        self.velocity_sensor    = np.ravel(self.sim_state[Sensors.VELOCITY_SENSOR])#/100.0 # Currently in cm - convert to m
        self.imu_sensor         = np.ravel(self.sim_state[Sensors.IMU_SENSOR])
        self.orientation_sensor = self.sim_state[Sensors.ORIENTATION_SENSOR]

    def get_state(self):
        return self.sim_state

    def get_camera(self):
        return self.camera_sensor

    def get_position(self):
        return self.position_sensor

    def get_world_velocity(self):
        return self.velocity_sensor

    def get_body_velocity(self):
        R = self.orientation_sensor
        world_vel = self.velocity_sensor
        body_vel = np.ravel(np.dot(R,world_vel)) # Rotate velocities into the body frame
        return body_vel

    def get_imu(self):
        self.imu_sensor[5] *= -1.0 # Yaw output seems to be backwards
        return self.imu_sensor

    def get_orientation(self):
        return self.orientation_sensor

    def get_euler(self):
        R = self.orientation_sensor
        euler = transforms3d.euler.mat2euler(R, 'rxyz')
        return euler

    def step_sim(self):
        if self.using_teleop:
            self.process_teleop()

        # Step holodeck simulator
        if not self.paused:
            self.sim_step += 1
            self.compute_control()
            self.sim_state, self.sim_reward, self.sim_terminal, self.sim_info = self.env.step(self.command)
            self.extract_sensor_data() # Get and store sensor data from state
            self.of_control.calc_optic_flow(np.copy(self.get_camera()))
            if self.plotting_states:
                t = self.sim_step*self.dt
                self.plotter.add_vector_measurement("position", self.get_position(), t)
                self.plotter.add_vector_measurement("velocity", self.get_body_velocity(), t)
                self.plotter.add_vector_measurement("orientation", self.get_euler(), t)
                self.plotter.add_vector_measurement("imu", self.get_imu(), t)
                self.plotter.add_vector_measurement("command", self.command, t)
                self.plotter.add_vector_measurement("vel_command", [self.vx_c, self.vy_c, self.yaw_c], t)
                self.plotter.update_plots()

        self.update_camera_window()

    def reset_sim(self):
        # Re-initialize commands
        self.set_command(0, 0, 0, 0)
        self.roll_c = 0.0
        self.pitch_c = 0.0
        self.yawrate_c = 0.0
        self.alt_c = 0.0
        self.vx_c = 0.0
        self.vy_c = 0.0
        self.yaw_c = 0.0

        # Re-initialize controllers
        self.vx_pid.reset()
        self.vy_pid.reset()

        # Reset the holodeck
        self.env.reset()

    def reset_commands(self):
        self.roll_c = 0.0
        self.pitch_c = 0.0
        self.yawrate_c = 0.0
        self.vx_c = 0.0
        self.vy_c = 0.0
        self.yaw_c = 0.0

        # Re-initialize controllers
        self.vx_pid.reset()
        self.vy_pid.reset()

    def exit_sim(self):
        sys.exit()
