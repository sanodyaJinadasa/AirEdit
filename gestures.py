# from utils import distance

# def is_drag(landmarks):
#     return distance(landmarks[8], landmarks[4]) > 40

# def is_resize(landmarks):
#     return distance(landmarks[8], landmarks[4]) < 40

# def is_fist(landmarks):
#     return landmarks[8][1] > landmarks[6][1] 



from utils import distance

PINCH_THRESHOLD = 40

def is_pinch(landmarks):
    return distance(landmarks[8], landmarks[4]) < PINCH_THRESHOLD

def is_fist(landmarks):
    return landmarks[8][1] > landmarks[6][1]
