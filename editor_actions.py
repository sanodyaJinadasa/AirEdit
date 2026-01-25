import cv2
import numpy as np

def move_image(image, x, y):
    image["x"] = x
    image["y"] = y

def resize_image(image, scale):
    image["scale"] *= scale

def rotate_image(image, angle):
    image["angle"] += angle



# ssuipo5
#SSL4
#22real
#23s
#ho