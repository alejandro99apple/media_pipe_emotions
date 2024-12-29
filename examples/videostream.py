import os
import sys
import cv2
sys.path.append(os.path.abspath(os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))))
from emotion_processor.main import EmotionRecognitionSystem

process = EmotionRecognitionSystem()

cap = cv2.VideoCapture(0)
cap.set(propId=3, value=1280)   
cap.set(propId=4, value=720)

if __name__ == "__main__":
    while True:
        ret, frame = cap.read()
        if not ret: # If the frame is not read correctly,
            break
        process.video_stream_processing(frame)
        cv2.imshow("Emotion Recognition", frame)
        t = cv2.waitKey(5)
        if t == 27: # ESC key to break
            break

    cap.release()
    cv2.destroyAllWindows()
