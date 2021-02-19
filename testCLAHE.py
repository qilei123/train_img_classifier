import numpy as np
import cv2
import cv2 as cv

test_img_dir = "/data1/qilei_chen/DATA/ROP_DATASET/images/00c9da2807dfa3115e7fb96c57da460b.png"

img = cv.imread(test_img_dir,0)
equ = cv.equalizeHist(img)
res = np.hstack((img,equ)) #stacking images side-by-side
cv.imwrite('equalhist.jpg',res)


img = cv.imread(test_img_dir,0)
# create a CLAHE object (Arguments are optional).
clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl1 = clahe.apply(img)
cv.imwrite('clahe_2.jpg',cl1)


bgr = cv2.imread(test_img_dir)

lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)

lab_planes = cv2.split(lab)

clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))

lab_planes[0] = clahe.apply(lab_planes[0])

lab = cv2.merge(lab_planes)

bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

cv.imwrite('clahe_bgr.jpg',bgr)