import numpy as np
import cv2
cap = cv2.VideoCapture('mv2_001.avi')
# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )
# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Nearly constant accel model
ts = 1.0/30.
A = np.eye(6)
A = A + np.diag([ts]*4, k = 2)


# Create some random colors
color = np.random.randint(0,255,(100,3))
# Take first frame and find corners in it
ret, img = cap.read()
ret = cv2.selectROI(img ,False)

old_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

mask = np.zeros_like(old_gray)
mask[ret[0]:ret[0]+ret[2], ret[1]:ret[1]+ret[3]] = True
cv2.imshow('mask', mask)

p0 = cv2.goodFeaturesToTrack(old_gray, mask = mask, **feature_params)
print(p0)

while(1):
    ret, img = cap.read()
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, img_gray, p0, None, **lk_params)

    # Select good points
    good_new = p1[st==1]
    good_old = p0[st==1]
    # draw the tracks
    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
        img = cv2.circle(img ,(a,b),5,color[i].tolist(),-1)
    img = cv2.add(img, mask)

    cv2.imshow('frame',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    # Now update the previous frame and previous points
    old_gray = img_gray.copy()
    p0 = good_new.reshape(-1,1,2)

cv2.destroyAllWindows()
cap.release()
