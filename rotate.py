import cv2
import pytesseract
import re

def rotate(image, center = None, scale = 1.0):
    angle=360-int(re.search('(?<=Rotate: )\d+', pytesseract.image_to_osd(image)).group(0))
    (h, w) = image.shape[:2]

    if center is None:
        center = (w / 2, h / 2)

    # Perform the rotation
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))

    return rotated

image = cv2.imread('deskewed/chep_tilted.png')
# grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# angle = determine_skew(grayscale)
rotated = rotate(image)
cv2.imwrite('rotated/chep_tilted.png', rotated)