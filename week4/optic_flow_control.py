import numpy as np
import cv2

class OpticFlowControl():

    def __init__(self, img_height, img_width, of_res = 20):
        '''
        Optical Flow Controller Constructor
        img_height - image height in pixels
        img_width - image width in pixels
        of_res - number of OF points along each axis
        '''
        self.prev_img = np.zeros((img_height,img_width))
        self.c_avg = np.zeros(2)
        self.l_avg = np.zeros(2)
        self.r_avg = np.zeros(2)
        self.d_avg = np.zeros(2)
        self.c_center = np.array([img_width//2, img_height//2])
        self.l_center = np.array([img_width//2 - 200, img_height//2])
        self.r_center = np.array([img_width//2 + 200, img_height//2])
        self.d_center = np.array([img_width//2, img_height//2 + 200])

        self.of_params = dict( winSize  = (15,15),
                               maxLevel = 3,
                               criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1, 0.03))

        self.feature_params = dict( maxCorners = 1000,
                       qualityLevel = 0.03,
                       minDistance = 7,
                       blockSize = 7 )

        # setup center mask
        self.center_mask = np.zeros((img_height, img_width), dtype=np.uint8)
        cv2.rectangle(self.center_mask, 
                (self.c_center[0] - 200, self.c_center[1] - 100),
                (self.c_center[0] + 200, self.c_center[1] + 100),
                color=255, thickness=cv2.FILLED)

        self.draw_center = np.zeros((img_height, img_width, 4), dtype=np.uint8)
        cv2.rectangle(self.draw_center, 
                (self.c_center[0] - 200, self.c_center[1] - 100),
                (self.c_center[0] + 200, self.c_center[1] + 100),
                color=(0, 0, 255, 0.5), thickness=1)

        # Setup right mask
        self.right_mask = np.zeros((img_height, img_width), dtype=np.uint8)
        cv2.rectangle(self.right_mask, 
                (self.r_center[0] - 100, self.r_center[1] - 150),
                (self.r_center[0] + 100, self.r_center[1] + 150),
                color=255, thickness=cv2.FILLED)

        self.draw_right = np.zeros((img_height, img_width, 4), dtype=np.uint8)
        cv2.rectangle(self.draw_right, 
                (self.r_center[0] - 100, self.r_center[1] - 150),
                (self.r_center[0] + 100, self.r_center[1] + 150),
                color=(0, 255, 0, 0.5), thickness=1)

        # Setup left mask
        self.left_mask = np.zeros((img_height, img_width), dtype=np.uint8)
        cv2.rectangle(self.left_mask, 
                (self.l_center[0] - 100, self.l_center[1] - 150),
                (self.l_center[0] + 100, self.l_center[1] + 150),
                color=255, thickness=cv2.FILLED)

        self.draw_left = np.zeros((img_height, img_width, 4), dtype=np.uint8)
        cv2.rectangle(self.draw_left, 
                (self.l_center[0] - 100, self.l_center[1] - 150),
                (self.l_center[0] + 100, self.l_center[1] + 150),
                color=(0, 255, 0, 0.5), thickness=1)

        # Setup down mask
        self.down_mask = np.zeros((img_height, img_width), dtype=np.uint8)
        cv2.rectangle(self.down_mask, 
                (self.d_center[0] - 100, self.d_center[1] - 100),
                (self.d_center[0] + 100, self.d_center[1] + 100),
                color=255, thickness=cv2.FILLED)

        self.draw_down = np.zeros((img_height, img_width, 4), dtype=np.uint8)
        cv2.rectangle(self.draw_down, 
                (self.d_center[0] - 100, self.d_center[1] - 100),
                (self.d_center[0] + 100, self.d_center[1] + 100),
                color=(255, 0, 0, 0.5), thickness=1)

        self.first = True

    def annotate(self, img):
        alpha = .5
        mask = np.zeros_like(img)

        # Add center flow
        p1_ = np.copy(self.c_flow[:,0,:])
        p1_ = map(tuple,p1_)
        p0_ = np.copy(self.p_center[:,0,:])
        p0_ = map(tuple,p0_) 

        for a,b in zip(p0_,p1_):
            mask_cof = cv2.arrowedLine(mask, a, b, (255, 255, 0), 1)

        mask_cavg = cv2.arrowedLine(mask, tuple(self.c_center), 
                tuple((self.c_center + self.c_avg).astype(np.int16)),
                (0, 255, 255), 1)

        # Add right flow
        p1_ = np.copy(self.r_flow[:,0,:])
        p1_ = map(tuple,p1_)
        p0_ = np.copy(self.p_right[:,0,:])
        p0_ = map(tuple,p0_) 

        for a,b in zip(p0_,p1_):
            mask_rof = cv2.arrowedLine(mask, a, b, (255, 255, 0), 1)

        mask_ravg = cv2.arrowedLine(mask, tuple(self.r_center), 
                tuple((self.r_center + self.r_avg).astype(np.int16)),
                (0, 255, 255), 1)

        # Add left flow
        p1_ = np.copy(self.l_flow[:,0,:])
        p1_ = map(tuple,p1_)
        p0_ = np.copy(self.p_left[:,0,:])
        p0_ = map(tuple,p0_) 

        for a,b in zip(p0_,p1_):
            mask_lof = cv2.arrowedLine(mask, a, b, (255, 255, 0), 1)

        mask_lavg = cv2.arrowedLine(mask, tuple(self.l_center), 
                tuple((self.l_center + self.l_avg).astype(np.int16)),
                (0, 255, 255), 1)

        # Add left flow
        p1_ = np.copy(self.d_flow[:,0,:])
        p1_ = map(tuple,p1_)
        p0_ = np.copy(self.p_down[:,0,:])
        p0_ = map(tuple,p0_) 

        for a,b in zip(p0_,p1_):
            mask_dof = cv2.arrowedLine(mask, a, b, (255, 255, 0), 1)

        mask_davg = cv2.arrowedLine(mask, tuple(self.d_center), 
                tuple((self.d_center + self.d_avg).astype(np.int16)),
                (0, 255, 255), 1)


        img = cv2.add(img, mask)
        img = cv2.add(img, self.draw_center)
        img = cv2.add(img, self.draw_right)
        img = cv2.add(img, self.draw_left)
        img = cv2.add(img, self.draw_down)
        return img
         

    def calc_optic_flow(self, img):
        if self.first:
            self.prev_img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
            self.first = False

        img_gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)

        self.p_center = cv2.goodFeaturesToTrack(self.prev_img, 
                mask= self.center_mask, **self.feature_params)
        self.p_right = cv2.goodFeaturesToTrack(self.prev_img, 
                mask= self.right_mask, **self.feature_params)
        self.p_left = cv2.goodFeaturesToTrack(self.prev_img, 
                mask= self.left_mask, **self.feature_params)
        self.p_down = cv2.goodFeaturesToTrack(self.prev_img, 
                mask= self.down_mask, **self.feature_params)

        # Add at least one point
        if self.p_center is None:
            self.p_center = np.array([[[self.c_center[0],self.c_center[1]]]],
                    dtype=np.float32)
        if self.p_right is None:
            self.p_right = np.array([[[self.r_center[0],self.r_center[1]]]],
                    dtype=np.float32)
        if self.p_left is None:
            self.p_left = np.array([[[self.l_center[0],self.l_center[1]]]],
                    dtype=np.float32)
        if self.p_down is None:
            self.p_down = np.array([[[self.d_center[0],self.d_center[1]]]],
                    dtype=np.float32)

        # calculate center flow
        self.c_flow, st, err = cv2.calcOpticalFlowPyrLK(self.prev_img, img_gray, 
                self.p_center, None, **self.of_params)
        self.c_avg = 20*np.mean(self.c_flow - self.p_center, axis=(0,1))

        # calculate right flow
        self.r_flow, st, err = cv2.calcOpticalFlowPyrLK(self.prev_img, img_gray, 
                self.p_right, None, **self.of_params)
        self.r_avg = 20*np.mean(self.r_flow - self.p_right, axis=(0,1))

        # calculate left flow
        self.l_flow, st, err = cv2.calcOpticalFlowPyrLK(self.prev_img, img_gray, 
                self.p_left, None, **self.of_params)
        self.l_avg = 20*np.mean(self.l_flow - self.p_left, axis=(0,1))

        # calculate down flow
        self.d_flow, st, err = cv2.calcOpticalFlowPyrLK(self.prev_img, img_gray, 
                self.p_down, None, **self.of_params)
        self.d_avg = 20*np.mean(self.d_flow - self.p_down, axis=(0,1))


        self.prev_img = img_gray


    def control_optic_flow(self, phi, theta, imu):
        roll_rate = imu[3] 
        pitch_rate = imu[4]
        yaw_rate = imu[5]

        kpvy = 10
        kpyaw = .2

        self.r_flow = self.r_flow - self.p_right
        self.l_flow = self.l_flow - self.p_left
        self.c_flow = self.c_flow - self.p_center

        # correct for yaw_rate
        self.r_flow[:,0,0] -= yaw_rate*(1. + ((self.p_right[:,0,0]-256)/256.)**2) 
        self.l_flow[:,0,0] -= yaw_rate*(1. + ((self.p_left[:,0,0]-256)/256.)**2)
        self.c_flow[:,0,0] -= yaw_rate*(1. + ((self.p_center[:,0,0]-256)/256.)**2)

        # correct for pitch_rate
        self.r_flow[:,0,1] += pitch_rate*(1. + ((self.p_right[:,0,1]-256)/256.)**2)
        self.l_flow[:,0,1] += pitch_rate*(1. + ((self.p_left[:,0,1]-256)/256.)**2)
        self.c_flow[:,0,1] += pitch_rate*(1. + ((self.p_center[:,0,1]-256)/256.)**2)
        
        self.r_avg = np.mean(self.r_flow, axis=(0,1))
        self.l_avg = np.mean(self.l_flow, axis=(0,1))
        self.c_avg = np.mean(self.c_flow, axis=(0,1))


        vy = -(self.r_avg[0] + self.l_avg[0])*kpvy
        yr = -(self.r_avg[0] + self.l_avg[0])*kpyaw
        print(yr)

        # vx, vy, yaw_rate, altitude
        return 250.0, vy, yr, 5.0



