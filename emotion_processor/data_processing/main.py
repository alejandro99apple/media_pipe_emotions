from emotion_processor.data_processing.eyes.eyes_processing import EyesPointsProcessing
from emotion_processor.data_processing.eyebrows.eyebrows_processing import EyebrowsPointsProcessing
from emotion_processor.data_processing.mouth.mouth_processing import MouthPointsProcessing
from emotion_processor.data_processing.nose.nose_processing import NosePointsProcessing

class PointsProcessing:
    def __init__(self):
        self.eyes = EyesPointsProcessing()
        self.eyebrows = EyebrowsPointsProcessing()
        self.mouth = MouthPointsProcessing()
        self.nose = NosePointsProcessing()

    def main(self, eyebrows_points: dict, eye_points:dict, mouth_points:dict, nose_points:dict):
        self.eyebrows.main(eyebrows_points)
        self.eyes.main(eye_points)
        self.mouth.main(mouth_points)
        self.nose.main(nose_points)


        return self.eyebrows, self.eyes, self.mouth, self.nose