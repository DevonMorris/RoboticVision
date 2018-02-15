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
        self.flow = np.zeros_like(self.prev_img)

        self.of_params = dict( winSize  = (15,15),
                               maxLevel = 3,
                               criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1, 0.03))

        x = np.linspace(0, img_width, num=of_res, dtype=np.int16)
        y = np.linspace(0, img_height, num=of_res, dtype=np.int16)
        xx, yy = np.meshgrid(x,y)
        xx = xx.flatten()
        yy = yy.flatten()
        self.of_pts = np.zeros((len(xx), 1, 2), dtype=np.float32)
        self.of_pts[:,0,0] = xx      
        self.of_pts[:,0,1] = yy     
        self.first = True

    def annotate(self, img):
        mask = np.zeros_like(img)

        # Make optical flow vectors
        p1_ = self.flow[:,0,:]
        p1_ = map(tuple,p1_)
        p0_ = self.of_pts[:,0,:]
        p0_ = map(tuple,p0_) 
        for a,b in zip(p0_,p1_):
            mask_of = cv2.arrowedLine(mask, a, b, (255, 255, 0), 1)
        img = cv2.add(img, mask_of)
        return img
         

    def calc_optic_flow(self, img):
        if self.first:
            self.prev_img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
            self.first = False

        img_gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
        self.flow, st, err = cv2.calcOpticalFlowPyrLK(self.prev_img, img_gray, 
                self.of_pts, None, **self.of_params)
        # Saturate/gate the optical flow
        diff = self.flow - self.of_pts
        norms = np.linalg.norm(diff, axis=2, keepdims=True)
        mask = (norms > 25).flatten()
        self.flow[mask] = 10*diff[mask]/norms[mask] + self.of_pts[mask]

        self.prev_image = img


    def control_optic_flow(self, img):
        return None



