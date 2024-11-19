from PIL import Image
import pytesseract as tess
import numpy as np
import cv2

tess.pytesseract.tesseract_cmd=r'/usr/local/Cellar/tesseract/4.1.1/bin/tesseract.exe'
filename= ''
#img = np.array(Image.open(filename))
#print(img)
#norm_img = np.zeros((img.shape[0], img.shape[1]))
#img = cv2.normalize(img, norm_img, 0, 255, cv2.NORM_MINMAX)
#img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)[1]
#img = cv2.GaussianBlur(img, (1, 1), 0)
path=r'/Users/abhaypatil/Downloads/demo.jpeg'
img=cv2.imread(path)

#convert to gray scale
gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#cv2.imshow('Original image',img)
#cv2.imshow('Gray image', gray)

#binary thresholding
#(thresh, blackAndWhiteImage) = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

adaptive_threshold= cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,91,11)

#cv2.imshow('Black white image', adaptive_threshold)
#cv2.imshow('Original image',img)
#cv2.imshow('gray image',gray)
cv2.waitKey(0)
cv2.destroyAllWindows()



text = tess.image_to_string(adaptive_threshold)
print(text)