import numpy as np
import warnings

class MouthPointsProcessing:
    def __init__(self):
        self.mouth: dict = {}


    def claculate_lip_arch(self, mouth_points):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', np.RankWarning)

            # Encuentra los puntos más externos de las cejas
            leftmost_point = mouth_points[0]
            rightmost_point = mouth_points[10]

            # Calcula el ángulo de inclinación
            delta_x = rightmost_point[0] - leftmost_point[0]
            delta_y = rightmost_point[1] - leftmost_point[1]
            angle = np.arctan2(delta_y, delta_x)

            # Rota los puntos para alinear la línea base con el eje horizontal
            rotation_matrix = np.array([
                [np.cos(-angle), -np.sin(-angle)],
                [np.sin(-angle), np.cos(-angle)]
            ])
            rotated_points = [np.dot(rotation_matrix, point) for point in mouth_points]

            # Extrae las coordenadas x e y de los puntos rotados
            x = [point[0] for point in rotated_points]
            y = [point[1] for point in rotated_points]

            # Ajuste polinomial
            z = np.polyfit(x, y, 2)
        return z[0] * -1000 # El coeficiente 'a' del polinomio y = ax^2 + bx + c
    

    def calculate_distances(self, mouth_points):
        mouth_opening_distance = np.linalg.norm(np.array(mouth_points['distances'][0]) - np.array(mouth_points['distances'][1]))
        mouth_reference_distance = np.linalg.norm(np.array(mouth_points['distances'][2]) - np.array(mouth_points['distances'][3]))
        return mouth_opening_distance, mouth_reference_distance
        

    def main(self, mouth_points: dict):
        self.mouth['upper_lip_arch'] = self.claculate_lip_arch(mouth_points['upper_lip'])
        self.mouth['lower_lip_arch'] = self.claculate_lip_arch(mouth_points['lower_lip'])
        self.mouth['mouth_opening_distance'], self.mouth['mouth_reference_distance'] = self.calculate_distances(mouth_points)
        return self.mouth