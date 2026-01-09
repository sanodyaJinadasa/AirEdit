import cv2
from hand_tracker import HandTracker
from gestures import is_drag, is_resize, is_fist
from image_manager import ImageManager
from undo_manager import UndoManager

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

hand = HandTracker()
images = ImageManager()
undo = UndoManager()

images.load_image("assets/image1.png", 200, 200)
images.load_image("assets/image2.png", 500, 300)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    landmarks = hand.get_landmarks(frame)

    if landmarks:
        if is_drag(landmarks):
            undo.save(images.images)
            images.images[0]["x"] = landmarks[8][0]
            images.images[0]["y"] = landmarks[8][1]

        if is_fist(landmarks):
            prev = undo.undo()
            if prev:
                images.images = prev

    # for img_data in images.images:
    #     x, y = img_data["x"], img_data["y"]
    #     frame[y:y+100, x:x+100] = img_data["img"][:100, :100]

    for img_data in images.images:
        img = img_data["img"]
        if img is not None:
            x, y = img_data["x"], img_data["y"]
            h, w = 100, 100 
            
            roi = frame[y:y+h, x:x+w]

            if img.shape[2] == 4:
                foreground = img[:h, :w, :3]
                alpha_mask = img[:h, :w, 3] / 255.0
                
                for c in range(0, 3):
                    frame[y:y+h, x:x+w, c] = (alpha_mask * foreground[:, :, c] +
                                            (1 - alpha_mask) * roi[:, :, c])
            else:
                frame[y:y+h, x:x+w] = img[:h, :w]

    cv2.imshow("Gesture Image Editor", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
