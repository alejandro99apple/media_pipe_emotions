from typing import Dict
#from emotion_processor.emotions_recognition.features.emotion_score import EmotionScore
from .emotions.surprise_score import SurpriseScore
#from .emotions.angry_score import AngryScore
#from .emotions.disgust_score import DisgustScore
from .emotions.sadness_score import SadnessScore
from .emotions.happiness_score import HappinessScore
#from .emotions.fear_score import FearScore


class EmotionRecognition:
    def __init__(self):
        self.emotions: Dict = {
            'surprise': SurpriseScore(),
            #'angry': AngryScore(),
            #'disgust': DisgustScore(),
            'sadness': SadnessScore(),
            'happiness': HappinessScore(),
            #'fear': FearScore(),
        }

    def recognize_emotion(self, processed_features:Dict) -> dict:
        scores = {}
        #self.emotions['surprise'].calculate_score(processed_features)
        #self.emotions['happiness'].calculate_score(processed_features)
        #self.emotions['sadness'].calculate_score(processed_features)

        scores = {
            'surprise': self.emotions['surprise'].calculate_score(processed_features),
            'sadness': self.emotions['sadness'].calculate_score(processed_features),
            'happiness': self.emotions['happiness'].calculate_score(processed_features),
        }
        return scores