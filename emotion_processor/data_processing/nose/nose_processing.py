import numpy as np

class NosePointsProcessing:
    def __init__(self):
        self.nose: dict = {}


    def calculate_distances(self, nose_points):
        top_distance = np.linalg.norm(np.array(nose_points['distances'][0]) - np.array(nose_points['distances'][1]))
        botton_distance = np.linalg.norm(np.array(nose_points['distances'][2]) - np.array(nose_points['distances'][3]))
        return botton_distance, top_distance

    def main(self,nose_points: dict):

        self.nose['topdistance'], self.nose['botton_distance'] = self.calculate_distances(nose_points)
        print(self.nose)
        return self.nose