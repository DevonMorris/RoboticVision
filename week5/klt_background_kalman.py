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

old_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

mask = np.zeros_like(old_gray)
mask[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]] = True

p0 = cv2.goodFeaturesToTrack(old_gray, mask = mask, **feature_params)

kf = cv2.KalmanFilter()
kf.transitionMatrix = A
kf.processNoiseCov = Q
kf.measurementMatrix = C
kf.measurementNoiseCov = R
kf.statePre = np.array([roi[0],  roi[1], 0., 0., 0., 0., 0., 0.], dtype=np.float32)
kf.errorCovPre = .1*np.eye(8, dtype=np.float32)
kf.controlMatrix = np.zeros((8,1), dtype=np.float32)

bg_sub = cv2.createBackgroundSubtractorMOG2(varThreshold = 40., detectShadows=False)
bg_sub.apply(img)

while(1):
    ret, img = cap.read()
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bg_mask = bg_sub.apply(img)
    bg_mask = cv2.morphologyEx(bg_mask, cv2.MORPH_OPEN, (7,7), iterations=3)
    bg_mask = cv2.morphologyEx(bg_mask, cv2.MORPH_CLOSE, (7,7), iterations=3)
    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, img_gray, p0, None, **lk_params)

    # Select good points
    good_new = p1[st==1]
    good_old = p0[st==1]

    x = np.mean(good_new, axis=0)
    flow = good_new - good_old
    flow = flow[np.linalg.norm(flow, axis=1) > .25]

    if flow.size != 0:
        flow = np.mean(flow, axis = 0)
    else:
        flow = np.array([0., 0.])

    kf.correct(np.array([x[0], x[1], 30*flow[0], 30*flow[1]], dtype=np.float32))
    # draw the tracks
    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
        # img = cv2.circle(img ,(a,b),5,color[i].tolist(),-1)

    img = cv2.rectangle( img,
            (int(roi[0]), int(roi[1])),
            (int(roi[0]+roi[2]), int(roi[1] + roi[3])),
            color=(255, 0, 0, 0.5), thickness = 2)


    cv2.imshow('frame',img)
    cv2.imshow('bg_sub', bg_mask)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    # Now update the previous frame and previous points
    old_gray = img_gray.copy()

    # find new feature points
    mask = np.zeros_like(old_gray)
    state = kf.statePost
    roi = (state[0], state[1],  
            roi[2], roi[3])
    mask[int(roi[1]):int(roi[1]+roi[3]),
            int(roi[0]):int(roi[0]+roi[2])] = True

    p0 = good_new.reshape(-1, 1, 2)
    p0_new = cv2.goodFeaturesToTrack(old_gray, mask = mask*bg_mask, **feature_params)
    if p0_new is not None:
        p0 = p0_new
        

    # predict forward kf
    kf.predict()

cv2.destroyAllWindows()
cap.release()
