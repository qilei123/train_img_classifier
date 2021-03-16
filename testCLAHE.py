import numpy as np
import cv2
import cv2 as cv
import os
import glob
'''
test_img_dir = "/data1/qilei_chen/DATA/erosive/images/0d8b52cc-2fbb-413e-b513-7037312175a1.jpg"

os.system("cp "+test_img_dir+" /data1/qilei_chen/DATA/erosive/temp/")

img = cv.imread(test_img_dir,0)
equ = cv.equalizeHist(img)
res = np.hstack((img,equ)) #stacking images side-by-side
cv.imwrite('/data1/qilei_chen/DATA/erosive/temp/equalhist.jpg',res)


img = cv.imread(test_img_dir,0)
# create a CLAHE object (Arguments are optional).
clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl1 = clahe.apply(img)
cv.imwrite('/data1/qilei_chen/DATA/erosive/temp/clahe_2.jpg',cl1)
'''
def equalhist_bgr(img,save_dir=None):
    pass

    #img = cv2.imread(test_img_dir)

    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

    # equalize the histogram of the Y channel
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])

    # convert the YUV image back to RGB format
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    if save_dir is not None:
        cv.imwrite(save_dir,img_output)
    return img_output

def clahe_bgr(img,save_dir=None):
    pass

    #bgr = cv2.imread(test_img_dir)

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)

    lab_planes = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(4,4))

    lab_planes[0] = clahe.apply(lab_planes[0])

    lab = cv2.merge(lab_planes)

    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    #cv.imwrite('/data1/qilei_chen/DATA/erosive/temp/clahe_bgr.jpg',bgr)
    if save_dir is not None:
        cv.imwrite(save_dir,bgr)
    return bgr

def main():
    file_dir_list = glob.glob(os.path.join("/data1/qilei_chen/DATA/erosive/images","*.jpg"))
    for file_dir in file_dir_list:
        img = cv2.imread(file_dir)
        equalhist_bgr(img,"/data1/qilei_chen/DATA/erosive/images_equalhist/")
        clahe_bgr(img,"/data1/qilei_chen/DATA/erosive/images_clahe/")

if __name__ == "__main__":
    main()