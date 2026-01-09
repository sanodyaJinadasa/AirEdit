import cv2

class ImageManager:
    def __init__(self):
        self.images = []

    def load_image(self, path, x, y):
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        self.images.append({
            "img": img,
            "x": x,
            "y": y,
            "scale": 1.0,
            "angle": 0
        })
