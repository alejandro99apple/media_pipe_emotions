import cv2
import numpy as np


class EmotionsVisualization:
    def __init__(self):
        self.emotion_colors = {
            'surprise': (255, 0, 255),
            'sadness': (186, 119, 4),
            'happiness': (0, 255, 127),
        }

    def main(self, emotions: dict, original_image: np.ndarray):
        for i, (emotion, score) in enumerate(emotions.items()):
            cv2.putText(original_image, emotion, (10, 30 + i * 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.emotion_colors[emotion], 1,cv2.LINE_AA)
            cv2.rectangle(original_image, (150, 15 + i * 40), (150 + int(score * 2.5), 35 + i * 40), self.emotion_colors[emotion],-1)
            cv2.rectangle(original_image, (150, 15 + i * 40), (400, 35 + i * 40), (255, 255, 255), 1)

        return original_image