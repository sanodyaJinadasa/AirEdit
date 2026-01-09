import math

def distance(p1, p2):
    return math.hypot(p1[0]-p2[0], p1[1]-p2[1])

def angle(p1, p2):
    return math.degrees(math.atan2(p2[1]-p1[1], p2[0]-p1[0]))
