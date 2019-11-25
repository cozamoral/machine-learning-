import cv2
import numpy as np
from matplotlib import pyplot as plt

image = cv2.imread('img.jpg')
(B, G, R) = image[100, 50]
print("R={}, G={}, B={}".format(R, G, B))

print(image)

# Convert BGR to HSV
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# extract a 100x100 pixel square ROI (Region of Interest) from the
# input image starting at x=320,y=60 at ending at x=420,y=160
roi = image[60:160, 320:420]
cv2.imwrite("copy.jpg", roi)

blurred = cv2.GaussianBlur(image, (11, 11), 0)
cv2.imwrite("blurred.jpg", blurred)

output = image.copy()
cv2.rectangle(output, (180, 10), (480, 440), (0, 0, 255), 2)
cv2.imshow("Rectangle", output)
cv2.imwrite("Rectangle.jpg", output)

img = cv2.imread('img.jpg', -1)
cv2.imshow('original', image)
cv2.waitKey(0)

img = cv2.imread('img.jpg', 0)
cv2.imshow('grey', img)
cv2.waitKey(0)

img = cv2.imread('img.jpg', 1)
cv2.imshow('color', img)
cv2.waitKey(0)

cv2.destroyAllWindows()

img = cv2.imread('img.jpg',cv2.IMREAD_GRAYSCALE)

plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
# to hide tick values on X and Y axis
plt.xticks([]), plt.yticks([])
plt.plot([200,300,400],[100,200,300],'c', linewidth = 5)
plt.show()

flags = [i for i in dir(cv2) if i.startswith('COLOR_')]
print(flags)

font = cv2.FONT_HERSHEY_SIMPLEX

cv2.putText(image, 'TEXT', (10, 200), font, 4, (255, 255, 255), 2, cv2.LINE_AA)

img = cv2.imread("img.jpg")
rows, cols, ch = img.shape
print("Height: ", rows)
print("Width: ", cols)


scaled_img = cv2.resize(img, (60, 60))

matrix_t = np.float32([[1, 0, -100], [0, 1, -30]])
translated_img = cv2.warpAffine(img, matrix_t, (cols, rows))


matrix_r = cv2.getRotationMatrix2D((cols/2, rows/2), 90, 0.5)
rotated_img = cv2.warpAffine(img, matrix_r, (cols, rows))


cv2.imshow("Original image", img)
cv2.imshow("Scaled image", scaled_img)
cv2.imshow("Translated image", translated_img)
cv2.imshow("Rotated image", rotated_img)
cv2.waitKey(0)
cv2.destroyAllWindows()