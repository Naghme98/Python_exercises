import cv2 as cv

img = cv.imread('img/thunder.jpg')
ret,thresh1 = cv.threshold(img,200,100,cv.THRESH_BINARY)
print(ret)
# creating pink one :)
# thresh1[:,:,2] = 0

thresh1[:,:,0] = 0
thresh1[:,:,1] = 0
img = img - thresh1
cv.imwrite('changed2.jpg', img)
