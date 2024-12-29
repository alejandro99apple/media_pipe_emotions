import numpy as np
from emotion_processor.face_mesh.main import FaceMeshMediaPipe


class EmotionRecognitionSystem:
    def __init__(self):
        self.face_mesh = FaceMeshMediaPipe()

    def video_stream_processing(self, face_image: np.ndarray):
        eye_points,eyebrown_points,mouth_points,nose_points,control_process, original_image = self.face_mesh.main_process(face_image)
        print(control_process)