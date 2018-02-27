import numpy as np
import cv2
cap = cv2.VideoCapture('mv2_001.avi')
# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = .3,
                       minDistance = 7,
                       blockSize = 7 )
# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.001))

# Nearly constant jerk model
ts = 1.0/30.
A = np.eye(8)
A = A + np.diag([ts]*6, k = 2)
A = A.astype(dtype=np.float32)

Q = np.array([[ts**7/252, 0., ts**6/72, 0., ts**5/30, 0., ts**4/24, 0.],
              [0., ts**7/252, 0., ts**6/72, 0., ts**5/30., 0., ts**4/24],
              [ts**6/72, 0., ts**5/20, 0., ts**4/8, 0., ts**3/6, 0.],
              [0., ts**6/72, 0., ts**5/20, 0., ts**4/8, 0., ts**3/6],
              [ts**5/30, 0., ts**4/8, 0., ts**3/3, 0., ts**2/2, 0.],
              [0., ts**5/30, 0., ts**4/8, 0., ts**3/3, 0., ts**2/2],
              [ts**4/24, 0., ts**3/6, 0., ts**2/2, 0., ts, 0.],
              [0., ts**4/24, 0., ts**3/6, 0., ts**2/2, 0., ts]])
Q = Q.astype(dtype=np.float32)

C = np.column_stack((np.eye(4), np.zeros((4,4))))
C = C.astype(dtype=np.float32)

R = .3*np.eye(4)
R[0,0] = 3
R[1,1] = 3
R = R.astype(dtype=np.float32)

# Create some random colors
color = np.random.randint(0,255,(100,3))
# Take first frame and find corners in it
ret, img = cap.read()
roi = cv2.selectROI(img ,False)

old_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
hsv_roi = old_hsv[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]]

mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0,180])

kf = cv2.KalmanFilter()
kf.transitionMatrix = A
kf.processNoiseCov = Q
kf.measurementMatrix = C
kf.measurementNoiseCov = R
kf.statePre = np.array([roi[0],  roi[1], 0., 0., 0., 0., 0., 0.], dtype=np.float32)
kf.errorCovPre = .1*np.eye(8, dtype=np.float32)
kf.controlMatrix = np.zeros((8,1), dtype=np.float32)

term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

while(1):
    ret, img = cap.read()
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject([img_hsv], [0], roi_hist, [0,180], 1)

    # apply meanshift to get new location
    ret, track_window = cv2.meanShift(dst, tuple(map(int, roi)), term_crit)

    kf.correct(np.array([track_window[0], track_window[1], 
        (track_window[0] - roi[0]),
        (track_window[1] - roi[1])], dtype=np.float32))

    img = cv2.rectangle( img,
            (int(roi[0]), int(roi[1])),
            (int(roi[0]+roi[2]), int(roi[1] + roi[3])),
            color=(255, 0, 0, 0.5), thickness = 2)


    cv2.imshow('frame',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    # Now update the previous frame and previous points
    old_hsv = img_hsv.copy()

    # find new feature points
    state = kf.statePost
    roi = (state[0], state[1],  
            roi[2], roi[3])

    # predict forward kf
    kf.predict()

cv2.destroyAllWindows()
cap.release()
