import numpy as np
from emotion_processor.data_processing.main import PointsProcessing
from emotion_processor.face_mesh.main import FaceMeshMediaPipe


class EmotionRecognitionSystem:
    def __init__(self):
        self.face_mesh = FaceMeshMediaPipe()
        self.data_processing = PointsProcessing()

    def video_stream_processing(self, face_image: np.ndarray):
        eye_points,eyebrow_points,mouth_points,nose_points,control_process, original_image = self.face_mesh.main_process(face_image, draw=True)
        
        if control_process:
            self.data_processing.main(eyebrow_points, eye_points, mouth_points, nose_points)
        else:
            raise Exception(f'No face Mesh')
        
            