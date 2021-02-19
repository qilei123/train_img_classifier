import numpy as np
import cv2 as cv

test_img_dir = "/data1/qilei_chen/DATA/ROP_DATASET/images/00c9da2807dfa3115e7fb96c57da460b.png"

img = cv.imread(test_img_dir)
equ = cv.equalizeHist(img)
res = np.hstack((img,equ)) #stacking images side-by-side
cv.imwrite('equalhist.jpg',res)


img = cv.imread(test_img_dir)
# create a CLAHE object (Arguments are optional).
clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl1 = clahe.apply(img)
cv.imwrite('clahe_2.jpg',cl1)