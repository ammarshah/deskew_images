import math
from typing import Tuple, Union

import cv2
import numpy as np

import pytesseract
import re

from deskew import determine_skew


def deskew(image: np.ndarray, angle: float, background: Union[int, Tuple[int, int, int]]) -> np.ndarray:
    old_width, old_height = image.shape[:2]
    angle_radian = math.radians(angle)
    width = abs(np.sin(angle_radian) * old_height) + abs(np.cos(angle_radian) * old_width)
    height = abs(np.sin(angle_radian) * old_width) + abs(np.cos(angle_radian) * old_height)

    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    rot_mat[1, 2] += (width - old_width) / 2
    rot_mat[0, 2] += (height - old_height) / 2
    return cv2.warpAffine(image, rot_mat, (int(round(height)), int(round(width))), borderValue=background)

def rotate(image):
    osd_rotated_image = pytesseract.image_to_osd(image)
    angle_to_be_rotated = re.search('(?<=Rotate: )\d+', osd_rotated_image).group(0)

    if angle_to_be_rotated == '0':
        rotated = image
    elif angle_to_be_rotated == '90':
        rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif angle_to_be_rotated == '180':
        rotated = cv2.rotate(image, cv2.ROTATE_180)
    elif angle_to_be_rotated == '270':
        rotated = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

    return rotated

image = cv2.imread('input_images/rombola.png')
grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
angle = determine_skew(grayscale)
deskewed = deskew(image, angle, (0, 0, 0))
rotated = rotate(deskewed)
cv2.imwrite('output_images/rombola.png', rotated)