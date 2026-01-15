import cv2
from hand_tracker import HandTracker
from gestures import is_pinch, is_fist
from image_manager import ImageManager
from undo_manager import UndoManager


dragging = False
active_image = 0


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
    if not ret:
        break
    frame = cv2.flip(frame, 1)  # mirror effect

    landmarks = hand.get_landmarks(frame)

    if landmarks:
        index_x, index_y = landmarks[8]  
        if landmarks:
            # draw a circle on index finger tip 
            cv2.circle(frame, (landmarks[8][0], landmarks[8][1]), 10, (0, 255, 0), -1)
            # draw a circle on thumb tip 
            cv2.circle(frame, (landmarks[4][0], landmarks[4][1]), 10, (0, 0, 255), -1)


        # ---------- drag ----------
        if is_pinch(landmarks):
            if not dragging:
                undo.save(images.images)  # save state once when drag starts
                dragging = True

            # boundary safe posistion
            frame_h, frame_w = frame.shape[:2]
            x = max(0, min(index_x - 50, frame_w - 100))
            y = max(0, min(index_y - 50, frame_h - 100))

            images.images[active_image]["x"] = x
            images.images[active_image]["y"] = y
        else:
            dragging = False

        if is_fist(landmarks):
            prev = undo.undo()
            if prev:
                images.images = prev

    
    for img_data in images.images:
        img = img_data["img"]
        if img is not None:
            x, y = img_data["x"], img_data["y"]
            h, w = 100, 100

            # make sure ROI is inside frame
            frame_h, frame_w = frame.shape[:2]
            if y + h > frame_h or x + w > frame_w:
                continue

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

    # ESC to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break


cap.release()
cv2.destroyAllWindows()
