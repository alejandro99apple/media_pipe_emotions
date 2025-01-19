import numpy as np
from emotion_processor.data_processing.main import PointsProcessing
from emotion_processor.face_mesh.main import FaceMeshMediaPipe
from emotion_processor.emotion_recognition.emotions.surprise_score import SurpriseScore
from emotion_processor.emotion_recognition.main import EmotionRecognition
from emotion_processor.visualization.visualization_main import EmotionsVisualization


class EmotionRecognitionSystem:
    def __init__(self):
        self.face_mesh = FaceMeshMediaPipe()
        self.data_processing = PointsProcessing()
        self.emotion_recognition = EmotionRecognition()
        self.emotions_visualization = EmotionsVisualization()

    def video_stream_processing(self, face_image: np.ndarray):
        eye_points, eyebrow_points, mouth_points, nose_points, control_process, original_image = self.face_mesh.main_process(face_image, draw=True)
        
        if control_process:
            processed_features = self.data_processing.main(eyebrow_points, eye_points, mouth_points, nose_points)
            emotions = self.emotion_recognition.recognize_emotion(processed_features)
            print(emotions)
            draw_emotions = self.emotions_visualization.main(emotions, original_image)

            return draw_emotions
        else:
            Exception(f'No face Mesh')
            return face_image
        
            